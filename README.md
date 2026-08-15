# deepagents-bot-42

A [deepagents](https://github.com/langchain-ai/deepagents) project deployable via LangGraph.

## Local development

Requires [Ollama](https://ollama.com) running locally with the model pulled:

```sh
ollama pull gemma4:12b-it-qat
uv sync
cp .env.example .env
uv run langgraph dev
```

## Deploy (self-hosted, via docker-compose)

```sh
uv run langgraph build -t deepagents-bot-42
IMAGE_NAME=deepagents-bot-42 docker compose up
```

The API is then served on `http://localhost:8123`.
