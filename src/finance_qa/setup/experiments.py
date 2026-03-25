"""Finance QA experiments — runs the chatbot on evaluation datasets and scores results."""
import uuid

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from utils.config import client
from src.finance_qa.agent.agent import create_chatbot

EXPERIMENT_MODELS = [
    "openai:gpt-4.1-mini",
    "openai:gpt-4.1",
    "anthropic:claude-haiku-4-5-20251001",
    "anthropic:claude-sonnet-4-6",
]


def _make_run_final_response(bot):
    """Create a final-response runner bound to the given chatbot."""
    def _run(inputs: dict) -> dict:
        question = inputs.get("question", "")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = bot.invoke({"messages": [HumanMessage(content=question)]}, config=config)
        for msg in reversed(result.get("messages", [])):
            if msg.type == "ai" and msg.content and not msg.tool_calls:
                return {"output": msg.content}
        return {"output": ""}
    return _run


def _make_run_chatbot(bot):
    """Create a full-messages runner bound to the given chatbot."""
    def _run(inputs: dict) -> dict:
        question = inputs.get("question", "")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = bot.invoke({"messages": [HumanMessage(content=question)]}, config=config)
        messages = result.get("messages", [])
        final_text = ""
        for msg in reversed(messages):
            if msg.type == "ai" and msg.content and not msg.tool_calls:
                final_text = msg.content
                break
        return {"messages": messages, "output": final_text}
    return _run


def _evaluate_has_response(outputs: dict, reference_outputs: dict) -> dict:
    """Check that the chatbot produced a non-empty final AI response."""
    for msg in reversed(outputs.get("messages", [])):
        if msg.type == "ai" and msg.content and not msg.tool_calls:
            return {"key": "has_response", "score": 1}
    return {"key": "has_response", "score": 0}


def _model_short_name(model_string: str) -> str:
    """Extract a short name from a provider:model string (e.g. 'openai:gpt-4.1-mini' -> 'gpt-4.1-mini')."""
    return model_string.split(":", 1)[-1] if ":" in model_string else model_string


def load_experiments() -> None:
    print("Loading experiments...")
    for model_string in EXPERIMENT_MODELS:
        short_name = _model_short_name(model_string)
        print(f"     - Running experiments with {short_name}...")
        chat_model = init_chat_model(model_string, temperature=0)
        bot = create_chatbot(chat_model)

        print(f"       - Final response experiment ({short_name})...")
        client.evaluate(
            _make_run_final_response(bot),
            data="Finance QA: Final Response",
            evaluators=[],  # helpfulness + answer_correctness are online evaluator rules
            experiment_prefix=f"finance-qa-final-response-{short_name}",
            num_repetitions=1,
            max_concurrency=3,
        )
        print(f"       - RAG citation experiment ({short_name})...")
        client.evaluate(
            _make_run_chatbot(bot),
            data="Finance QA: RAG Citation",
            evaluators=[_evaluate_has_response],
            experiment_prefix=f"finance-qa-rag-citation-{short_name}",
            num_repetitions=1,
            max_concurrency=3,
        )
    print("Experiments loaded successfully.")


if __name__ == "__main__":
    load_experiments()
