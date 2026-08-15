import re
from pathlib import Path

import pytest
from langgraph_sdk import Auth

from security.auth import (
    RESERVED_METADATA_KEYS,
    VALID_TOKENS,
    auth,
    get_current_user,
    on_assistant_update,
    on_cron_create,
    on_cron_update,
    on_store_get,
    on_store_list_namespaces,
    on_store_put,
    on_thread_create,
    on_thread_create_run,
    on_thread_read,
    on_thread_search,
    on_thread_update,
)

pytestmark = pytest.mark.anyio


class _FakeUser:
    def __init__(self, identity: str) -> None:
        self.identity = identity


def _ctx(identity: str, resource: str = "threads", action: str = "create") -> Auth.types.AuthContext:
    return Auth.types.AuthContext(
        permissions=[],
        user=_FakeUser(identity),
        resource=resource,
        action=action,
    )


# ---------------------------------------------------------------------------
# Baseline authentication behavior
# ---------------------------------------------------------------------------


async def test_missing_authorization_header_raises_401() -> None:
    with pytest.raises(Auth.exceptions.HTTPException) as exc_info:
        await get_current_user(None)
    assert exc_info.value.status_code == 401


async def test_malformed_authorization_header_raises_401() -> None:
    with pytest.raises(Auth.exceptions.HTTPException) as exc_info:
        await get_current_user("Bearer")
    assert exc_info.value.status_code == 401


async def test_wrong_scheme_raises_401() -> None:
    with pytest.raises(Auth.exceptions.HTTPException) as exc_info:
        await get_current_user("Basic user1-token")
    assert exc_info.value.status_code == 401


async def test_invalid_token_raises_401() -> None:
    with pytest.raises(Auth.exceptions.HTTPException) as exc_info:
        await get_current_user("Bearer not-a-real-token")
    assert exc_info.value.status_code == 401


async def test_valid_token_returns_identity() -> None:
    user = await get_current_user("Bearer user1-token")
    assert user == {"identity": "user1"}


# ---------------------------------------------------------------------------
# Attack 1: client-supplied metadata poisoning
# ---------------------------------------------------------------------------


async def test_reserved_metadata_key_is_rejected_on_create() -> None:
    """A client trying to set metadata.owner directly must be denied, not honored."""
    value = {"metadata": {"owner": "victim-user"}}
    with pytest.raises(Auth.exceptions.HTTPException) as exc_info:
        await on_thread_create(_ctx("attacker"), value)
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize("reserved_key", sorted(RESERVED_METADATA_KEYS))
async def test_each_reserved_key_is_rejected(reserved_key: str) -> None:
    value = {"metadata": {reserved_key: "attacker-controlled"}}
    with pytest.raises(Auth.exceptions.HTTPException) as exc_info:
        await on_thread_create(_ctx("attacker"), value)
    assert exc_info.value.status_code == 400


async def test_non_reserved_metadata_survives_alongside_pinned_owner() -> None:
    """Legitimate client metadata is preserved; only ownership is server-controlled."""
    value = {"metadata": {"project": "roadmap"}}
    await on_thread_create(_ctx("user1"), value)
    assert value["metadata"]["project"] == "roadmap"
    assert value["metadata"]["owner"] == "user1"


# ---------------------------------------------------------------------------
# Attack 2: resource planting (cross-tenant prompt injection via metadata)
# ---------------------------------------------------------------------------


async def test_created_resource_owner_is_always_the_actual_creator() -> None:
    """Even with no malicious metadata, ownership is always the caller's own identity."""
    value: dict = {}
    await on_thread_create(_ctx("attacker"), value)
    assert value["metadata"]["owner"] == "attacker"


async def test_planted_resource_does_not_pass_victims_read_filter() -> None:
    """A resource owned by 'attacker' must not satisfy a filter scoped to 'victim'."""
    value: dict = {}
    await on_thread_create(_ctx("attacker"), value)
    victim_filter = await on_thread_read(_ctx("victim"), {})
    assert victim_filter == {"owner": "victim"}
    assert value["metadata"]["owner"] != victim_filter["owner"]


async def test_search_filter_is_always_scoped_to_caller_regardless_of_client_filter() -> None:
    """A client cannot use its own search-time metadata filter to see another owner's data."""
    value = {"metadata": {"owner": "victim"}}  # attacker-supplied search filter, not stored data
    filters = await on_thread_search(_ctx("attacker"), value)
    assert filters == {"owner": "attacker"}


# ---------------------------------------------------------------------------
# Attack 3: ownership rewrite on update / create_run
# ---------------------------------------------------------------------------


async def test_thread_update_cannot_reparent_owner() -> None:
    value = {"metadata": {"owner": "attacker"}}
    with pytest.raises(Auth.exceptions.HTTPException):
        await on_thread_update(_ctx("attacker", action="update"), value)


async def test_thread_create_run_cannot_reparent_owner() -> None:
    value = {"metadata": {"owner": "attacker"}}
    with pytest.raises(Auth.exceptions.HTTPException):
        await on_thread_create_run(_ctx("attacker", action="create_run"), value)


async def test_assistant_update_cannot_reparent_owner() -> None:
    value = {"metadata": {"owner": "attacker"}}
    with pytest.raises(Auth.exceptions.HTTPException):
        await on_assistant_update(_ctx("attacker", "assistants", "update"), value)


async def test_thread_update_pins_owner_when_metadata_is_clean() -> None:
    value = {"metadata": {"title": "renamed"}}
    await on_thread_update(_ctx("user1", action="update"), value)
    assert value["metadata"]["owner"] == "user1"


async def test_cron_update_cannot_target_another_users_cron() -> None:
    """CronsUpdate has no ownership field to rewrite; the filter must still scope by user_id."""
    filters = await on_cron_update(_ctx("user1", "crons", "update"), {"cron_id": "abc"})
    assert filters == {"user_id": "user1"}


async def test_cron_create_pins_user_id_ignoring_client_value() -> None:
    value = {"user_id": "attacker-wants-someone-else"}
    await on_cron_create(_ctx("user1", "crons", "create"), value)
    assert value["user_id"] == "user1"


# ---------------------------------------------------------------------------
# Attack 4: complete handler coverage
# ---------------------------------------------------------------------------


EXPECTED_COVERAGE = {
    ("threads", "create"),
    ("threads", "read"),
    ("threads", "update"),
    ("threads", "delete"),
    ("threads", "search"),
    ("threads", "create_run"),
    ("assistants", "create"),
    ("assistants", "read"),
    ("assistants", "update"),
    ("assistants", "delete"),
    ("assistants", "search"),
    ("crons", "create"),
    ("crons", "read"),
    ("crons", "update"),
    ("crons", "delete"),
    ("crons", "search"),
    ("store", "put"),
    ("store", "get"),
    ("store", "search"),
    ("store", "delete"),
    ("store", "list_namespaces"),
}


def test_every_supported_action_has_a_specific_handler() -> None:
    assert EXPECTED_COVERAGE <= set(auth._handlers.keys())


def test_global_fallback_handler_is_registered() -> None:
    assert len(auth._global_handlers) == 1


async def test_global_fallback_denies_by_default() -> None:
    from security.auth import deny_unhandled

    result = await deny_unhandled(_ctx("someone", "unknown-resource", "unknown-action"), {})
    assert result is False


# ---------------------------------------------------------------------------
# Attack 5: no `assert` used for authorization decisions
# ---------------------------------------------------------------------------


def test_auth_module_contains_no_assert_statements() -> None:
    """`assert` is stripped under `python -O`, which would silently disable checks."""
    source = Path("security/auth.py").read_text()
    assert not re.search(r"(?m)^\s*assert\b", source)


# ---------------------------------------------------------------------------
# Attack 6: store namespace handling
# ---------------------------------------------------------------------------


async def test_store_missing_namespace_is_scoped_to_caller() -> None:
    value: dict = {}
    await on_store_get(_ctx("user1", "store", "get"), value)
    assert value["namespace"] == ("user1",)


async def test_store_none_namespace_is_scoped_to_caller() -> None:
    """list_namespaces explicitly allows namespace=None."""
    value = {"namespace": None}
    await on_store_list_namespaces(_ctx("user1", "store", "list_namespaces"), value)
    assert value["namespace"] == ("user1",)


async def test_store_empty_namespace_is_scoped_to_caller() -> None:
    value = {"namespace": ()}
    await on_store_put(_ctx("user1", "store", "put"), value)
    assert value["namespace"] == ("user1",)


async def test_store_other_users_namespace_is_nested_under_caller() -> None:
    """A client trying to address another user's top-level namespace gets sandboxed instead."""
    value = {"namespace": ("victim", "secrets")}
    await on_store_get(_ctx("attacker", "store", "get"), value)
    assert value["namespace"] == ("attacker", "victim", "secrets")


async def test_store_malformed_namespace_type_is_rejected() -> None:
    """A bare string would silently explode into per-character segments via tuple(); reject it."""
    value = {"namespace": "not-a-tuple"}
    with pytest.raises(Auth.exceptions.HTTPException) as exc_info:
        await on_store_get(_ctx("user1", "store", "get"), value)
    assert exc_info.value.status_code == 400


async def test_store_namespace_with_empty_segment_is_rejected() -> None:
    value = {"namespace": ("user1", "")}
    with pytest.raises(Auth.exceptions.HTTPException) as exc_info:
        await on_store_get(_ctx("user1", "store", "get"), value)
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Attack 7: identity provenance
# ---------------------------------------------------------------------------


async def test_identity_is_the_stable_id_not_the_mutable_display_name() -> None:
    user = await get_current_user("Bearer user1-token")
    assert user["identity"] == VALID_TOKENS["user1-token"]["id"]
    assert user["identity"] != VALID_TOKENS["user1-token"]["name"]


# ---------------------------------------------------------------------------
# Attack 8: malformed input
# ---------------------------------------------------------------------------


async def test_null_metadata_does_not_crash() -> None:
    """`{"metadata": null}` must not raise AttributeError from `.update()` on None."""
    value = {"metadata": None}
    await on_thread_create(_ctx("user1"), value)
    assert value["metadata"] == {"owner": "user1"}


async def test_non_dict_metadata_is_rejected() -> None:
    value = {"metadata": "not-a-dict"}
    with pytest.raises(Auth.exceptions.HTTPException) as exc_info:
        await on_thread_create(_ctx("user1"), value)
    assert exc_info.value.status_code == 400


async def test_too_many_metadata_keys_is_rejected() -> None:
    value = {"metadata": {f"key{i}": i for i in range(51)}}
    with pytest.raises(Auth.exceptions.HTTPException) as exc_info:
        await on_thread_create(_ctx("user1"), value)
    assert exc_info.value.status_code == 400


async def test_oversized_metadata_is_rejected() -> None:
    value = {"metadata": {"blob": "x" * 10_000}}
    with pytest.raises(Auth.exceptions.HTTPException) as exc_info:
        await on_thread_create(_ctx("user1"), value)
    assert exc_info.value.status_code == 400
