"""Hardened authentication and authorization for the LangGraph server.

Threat model: `value` passed into every `@auth.on` handler below is the raw,
client-supplied request payload. It is untrusted input. In particular,
`value["metadata"]` is NOT server state -- it is whatever the caller typed
into their SDK/HTTP request, so it must never be merged directly into the
metadata we use for authorization filters. See inline `THREAT #n` comments
for which control defends against which attack; the numbering matches the
review checklist this file was written against.

Ported from https://github.com/langchain-ai/deepagents-demo (src/security/auth.py).
"""

import json
from typing import Any

from langgraph_sdk import Auth

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

# This is our toy user database. Do not do this in production.
VALID_TOKENS = {
    "user1-token": {"id": "user1", "name": "Alice"},
    "user2-token": {"id": "user2", "name": "Bob"},
}

auth = Auth()


@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    """Verify the bearer token and return a stable, issuer-verified identity."""
    if not authorization:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Missing authorization header"
        )
    try:
        scheme, token = authorization.split()
    except ValueError:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Invalid authorization header"
        ) from None
    if scheme.lower() != "bearer":
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Invalid authorization scheme"
        )

    user_record = VALID_TOKENS.get(token)
    if user_record is None:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token")

    # THREAT #7 (identity provenance / account-takeover-by-rename):
    # `identity` must come from a stable, server-controlled, issuer-verified
    # claim -- here that's the opaque `id` looked up from our own token
    # table, never `user_record["name"]`. All ownership metadata below is
    # keyed off this value, so it must be immutable for the life of the
    # account. A display name or email is often user-editable (a user
    # renames themselves, or an IdP lets them change their email); if that
    # were used as `identity`, a rename could cause a *new* identity string
    # to inherit every resource previously owned by whoever held that name
    # before -- effectively an account takeover via rename. In a real
    # JWT-based deployment, decode the token and pin to the issuer's
    # immutable subject claim, not a mutable profile field:
    #
    #     payload = jwt.decode(token, PUBLIC_KEY, algorithms=["RS256"], audience=AUD)
    #     return {"identity": payload["sub"], "display_name": payload.get("name")}
    return {"identity": user_record["id"]}


# ---------------------------------------------------------------------------
# Metadata hardening helpers (threads / assistants / runs)
# ---------------------------------------------------------------------------

RESERVED_METADATA_KEYS = frozenset({"owner", "allowed_users", "org_id"})
"""Metadata keys that participate in authorization filters.

These may only ever be written by our own handlers below, never by a client
-- otherwise a client could plant a value (e.g. `metadata.owner =
"other-user"`) that later causes the resource to be returned by another
tenant's read/search filter. (THREAT #1, #2)
"""

MAX_METADATA_KEYS = 50
MAX_METADATA_BYTES = 8_000


def _validate_client_metadata(raw: dict[str, Any]) -> None:
    """Validate client-supplied metadata in place; raise on anything invalid.

    Defends against:
      - THREAT #8 (malformed input): reject anything that isn't a JSON
        object, and bound both key count and serialized size so a client
        can't blow up storage with an oversized blob.
      - THREAT #1 (metadata poisoning): reserved keys are rejected here,
        before any handler below has a chance to merge client metadata into
        the stored/filterable metadata.

    Deliberately does NOT copy or return a new dict -- it validates `raw`
    in place. `langgraph_api` builds the `value` passed to our handlers by
    aliasing its own local `metadata` variable (e.g. `{"metadata": metadata,
    ...}`) and, after our handler returns, reads that *local* variable --
    not `value["metadata"]` -- to persist the resource. `handle_event` never
    returns the mutated `value` back to the caller. So a handler that does
    `value["metadata"] = new_dict` (rather than mutating the existing dict
    object in place) silently discards its own changes: the framework still
    persists the original, unmodified metadata. Concretely this means
    `_pin_owner`'s `owner` stamp would never actually land in storage,
    every subsequent owner-scoped read/search/create_run filter would fail
    to match the resource, and the resource would become permanently
    inaccessible (a `NotFoundError`) even to its own creator.
    """
    if not isinstance(raw, dict):
        raise Auth.exceptions.HTTPException(
            status_code=400, detail="metadata must be a JSON object"
        )
    if len(raw) > MAX_METADATA_KEYS:
        raise Auth.exceptions.HTTPException(
            status_code=400, detail="too many metadata keys"
        )
    try:
        encoded_size = len(json.dumps(raw))
    except TypeError:
        raise Auth.exceptions.HTTPException(
            status_code=400, detail="metadata must be JSON-serializable"
        ) from None
    if encoded_size > MAX_METADATA_BYTES:
        raise Auth.exceptions.HTTPException(
            status_code=400, detail="metadata payload too large"
        )

    reserved_present = RESERVED_METADATA_KEYS & raw.keys()
    if reserved_present:
        raise Auth.exceptions.HTTPException(
            status_code=400,
            detail=(
                "metadata keys are reserved and may not be set by the "
                f"client: {sorted(reserved_present)}"
            ),
        )


def _pin_owner(value: dict[str, Any], ctx: Auth.types.AuthContext) -> dict[str, Any]:
    """Force `metadata["owner"]` to the caller's own identity.

    Called on EVERY mutating action (create, update, create_run) -- not just
    create -- so that:
      - THREAT #2 (resource planting): a client cannot create a resource
        whose metadata claims a different owner, which would otherwise let
        it surface in that victim's read/search results and get pulled into
        an agent session running with the victim's credentials (cross-tenant
        prompt injection).
      - THREAT #3 (ownership rewrite on update): a client cannot PATCH an
        existing resource's metadata to re-parent it to another user, since
        we overwrite (not merge) the `owner` key on every write path, using
        the caller's own authenticated identity rather than anything in the
        request body.

    Mutates `value["metadata"]` in place -- never reassigns it -- see
    `_validate_client_metadata` for why that distinction matters here.
    """
    if value.get("metadata") is None:
        value["metadata"] = {}
    metadata = value["metadata"]
    _validate_client_metadata(metadata)
    metadata["owner"] = ctx.user.identity
    return metadata


def _owner_filter(ctx: Auth.types.AuthContext) -> Auth.types.FilterType:
    """Scope reads/searches/deletes to resources owned by the caller."""
    return {"owner": ctx.user.identity}


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


@auth.on.threads.create
async def on_thread_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.create.value,
) -> None:
    """Stamp ownership on new threads; never trust client-sent metadata."""
    _pin_owner(value, ctx)


@auth.on.threads.read
async def on_thread_read(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.read.value,
) -> Auth.types.FilterType:
    """Only allow reading threads owned by the caller."""
    return _owner_filter(ctx)


@auth.on.threads.update
async def on_thread_update(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.update.value,
) -> Auth.types.FilterType:
    """Re-pin ownership on update (THREAT #3) and restrict to own threads."""
    _pin_owner(value, ctx)
    return _owner_filter(ctx)


@auth.on.threads.delete
async def on_thread_delete(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.delete.value,
) -> Auth.types.FilterType:
    """Only allow deleting threads owned by the caller."""
    return _owner_filter(ctx)


@auth.on.threads.search
async def on_thread_search(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.search.value,
) -> Auth.types.FilterType:
    """Only surface the caller's own threads in search results.

    Note: `value["metadata"]` here is the CLIENT'S search filter (narrows
    which of *their* results come back), not data being written -- it is
    always AND-ed with the filter we return below, so a client cannot use it
    to escape the owner scoping (THREAT #2).
    """
    return _owner_filter(ctx)


@auth.on.threads.create_run
async def on_thread_create_run(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.create_run.value,
) -> Auth.types.FilterType:
    """Re-pin run ownership (THREAT #3); require the target thread be the caller's own."""
    _pin_owner(value, ctx)
    return _owner_filter(ctx)


# ---------------------------------------------------------------------------
# Assistants
# ---------------------------------------------------------------------------


@auth.on.assistants.create
async def on_assistant_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.create.value,
) -> None:
    """Stamp ownership on new assistants; never trust client-sent metadata."""
    _pin_owner(value, ctx)


@auth.on.assistants.read
async def on_assistant_read(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.read.value,
) -> Auth.types.FilterType:
    """Only allow reading assistants owned by the caller."""
    return _owner_filter(ctx)


@auth.on.assistants.update
async def on_assistant_update(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.update.value,
) -> Auth.types.FilterType:
    """Re-pin ownership on update (THREAT #3) and restrict to own assistants."""
    _pin_owner(value, ctx)
    return _owner_filter(ctx)


@auth.on.assistants.delete
async def on_assistant_delete(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.delete.value,
) -> Auth.types.FilterType:
    """Only allow deleting assistants owned by the caller."""
    return _owner_filter(ctx)


@auth.on.assistants.search
async def on_assistant_search(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.search.value,
) -> Auth.types.FilterType:
    """Only surface the caller's own assistants in search results."""
    return _owner_filter(ctx)


# ---------------------------------------------------------------------------
# Crons
# ---------------------------------------------------------------------------
# Unlike threads/assistants/runs, `CronsCreate` in the installed
# `langgraph_sdk` does not declare a `metadata` field at all -- it declares a
# first-class `user_id: str | None` field for exactly this purpose, and
# `CronsUpdate`/`CronsRead`/`CronsDelete`/`CronsSearch` carry no ownership
# field whatsoever. So crons are scoped on `user_id`, not `metadata.owner`.
# This is a deliberate, schema-grounded choice, not a guess -- verify against
# your deployed server version if you rely on this.


def _cron_owner_filter(ctx: Auth.types.AuthContext) -> Auth.types.FilterType:
    return {"user_id": ctx.user.identity}


@auth.on.crons.create
async def on_cron_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.crons.create.value,
) -> None:
    """Stamp ownership on new crons; never trust a client-sent `user_id` (THREAT #1, #2)."""
    value["user_id"] = ctx.user.identity


@auth.on.crons.read
async def on_cron_read(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.crons.read.value,
) -> Auth.types.FilterType:
    """Only allow reading crons owned by the caller."""
    return _cron_owner_filter(ctx)


@auth.on.crons.update
async def on_cron_update(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.crons.update.value,
) -> Auth.types.FilterType:
    """Restrict updates to the caller's own crons.

    `CronsUpdate` carries no `user_id`/ownership field to re-pin -- the
    update payload structurally cannot re-parent a cron (THREAT #3 doesn't
    apply here) -- but we still scope which cron the update may target.
    """
    return _cron_owner_filter(ctx)


@auth.on.crons.delete
async def on_cron_delete(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.crons.delete.value,
) -> Auth.types.FilterType:
    """Only allow deleting crons owned by the caller."""
    return _cron_owner_filter(ctx)


@auth.on.crons.search
async def on_cron_search(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.crons.search.value,
) -> Auth.types.FilterType:
    """Only surface the caller's own crons in search results."""
    return _cron_owner_filter(ctx)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

MAX_NAMESPACE_DEPTH = 20


def _scoped_namespace(
    ctx: Auth.types.AuthContext, value: dict[str, Any]
) -> tuple[str, ...]:
    """Force every store namespace under the caller's own identity prefix.

    Defends against THREAT #6 (store namespace handling): the client-supplied
    `namespace` can be missing, `None` (explicitly valid for
    `list_namespaces`), an empty tuple, or a malformed type (e.g. a bare
    string, which `tuple(...)` would silently explode into one-character
    segments). Any of these would otherwise cause an incidental
    `IndexError`/`KeyError`/`TypeError` in code that reads `namespace[0]`
    without checking -- or worse, silently apply the wrong scope. We
    validate explicitly and fail closed with a 400 instead.

    Mutates `value["namespace"]` in place, following the pattern documented
    by the SDK itself, so the server applies the corrected, scoped namespace
    to the actual operation: the client's requested segments are nested
    *under* their own identity, so they can never read/write/list another
    user's prefix no matter what they pass in.
    """
    raw = value.get("namespace")
    if raw is None:
        namespace: tuple[str, ...] = ()
    elif isinstance(raw, (tuple, list)):
        namespace = tuple(raw)
    else:
        raise Auth.exceptions.HTTPException(
            status_code=400, detail="namespace must be a list/tuple of strings"
        )

    if not all(isinstance(segment, str) and segment for segment in namespace):
        raise Auth.exceptions.HTTPException(
            status_code=400, detail="namespace segments must be non-empty strings"
        )
    if len(namespace) > MAX_NAMESPACE_DEPTH:
        raise Auth.exceptions.HTTPException(status_code=400, detail="namespace too deep")

    if not namespace or namespace[0] != ctx.user.identity:
        namespace = (ctx.user.identity, *namespace)

    value["namespace"] = namespace
    return namespace


@auth.on.store.put
async def on_store_put(
    ctx: Auth.types.AuthContext, value: Auth.types.on.store.put.value
) -> None:
    """Scope store writes to the caller's own namespace."""
    _scoped_namespace(ctx, value)


@auth.on.store.get
async def on_store_get(
    ctx: Auth.types.AuthContext, value: Auth.types.on.store.get.value
) -> None:
    """Scope store reads to the caller's own namespace."""
    _scoped_namespace(ctx, value)


@auth.on.store.search
async def on_store_search(
    ctx: Auth.types.AuthContext, value: Auth.types.on.store.search.value
) -> None:
    """Scope store searches to the caller's own namespace."""
    _scoped_namespace(ctx, value)


@auth.on.store.delete
async def on_store_delete(
    ctx: Auth.types.AuthContext, value: Auth.types.on.store.delete.value
) -> None:
    """Scope store deletes to the caller's own namespace."""
    _scoped_namespace(ctx, value)


@auth.on.store.list_namespaces
async def on_store_list_namespaces(
    ctx: Auth.types.AuthContext, value: Auth.types.on.store.list_namespaces.value
) -> None:
    """Scope namespace listing to the caller's own namespace prefix."""
    _scoped_namespace(ctx, value)


# ---------------------------------------------------------------------------
# Global fallback (THREAT #4: complete handler coverage)
# ---------------------------------------------------------------------------
# Every resource/action pair in the supported-actions table (threads:
# create/read/update/delete/search/create_run; assistants: *; crons: *;
# store: put/get/search/delete/list_namespaces) has an explicit handler
# above. This global handler only fires for anything NOT covered -- e.g. a
# resource/action added by a future SDK version -- and fails closed rather
# than silently letting it through.


@auth.on
async def deny_unhandled(ctx: Auth.types.AuthContext, value: Auth.types.on.value) -> bool:
    """Deny by default; every known resource/action has a specific handler above."""
    return False


# ---------------------------------------------------------------------------
# Coverage table: resource/action -> handler -> policy
# ---------------------------------------------------------------------------
#
# | Resource   | Action          | Handler                     | Policy                                        |
# |------------|------------------|------------------------------|------------------------------------------------|
# | threads    | create           | on_thread_create             | pin metadata.owner                            |
# | threads    | read             | on_thread_read                | filter: metadata.owner == caller               |
# | threads    | update           | on_thread_update              | re-pin metadata.owner + filter                 |
# | threads    | delete           | on_thread_delete              | filter: metadata.owner == caller               |
# | threads    | search           | on_thread_search              | filter: metadata.owner == caller               |
# | threads    | create_run       | on_thread_create_run          | re-pin metadata.owner + filter                 |
# | assistants | create           | on_assistant_create           | pin metadata.owner                            |
# | assistants | read             | on_assistant_read             | filter: metadata.owner == caller               |
# | assistants | update           | on_assistant_update           | re-pin metadata.owner + filter                 |
# | assistants | delete           | on_assistant_delete           | filter: metadata.owner == caller               |
# | assistants | search           | on_assistant_search           | filter: metadata.owner == caller               |
# | crons      | create           | on_cron_create                | pin user_id                                    |
# | crons      | read             | on_cron_read                  | filter: user_id == caller                      |
# | crons      | update           | on_cron_update                | filter: user_id == caller                      |
# | crons      | delete           | on_cron_delete                | filter: user_id == caller                      |
# | crons      | search           | on_cron_search                | filter: user_id == caller                      |
# | store      | put              | on_store_put                  | namespace forced under (caller, ...)           |
# | store      | get              | on_store_get                  | namespace forced under (caller, ...)           |
# | store      | search           | on_store_search               | namespace forced under (caller, ...)           |
# | store      | delete           | on_store_delete               | namespace forced under (caller, ...)           |
# | store      | list_namespaces  | on_store_list_namespaces      | namespace forced under (caller, ...)           |
# | *          | *                | deny_unhandled                 | deny (fail closed)                             |
