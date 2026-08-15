from langchain.chat_models import init_chat_model

from deepagents import create_deep_agent

model = init_chat_model("ollama:gemma4:12b-it-qat")

agent = create_deep_agent(
    model=model,
    system_prompt="You are a helpful deep agent.",
)
