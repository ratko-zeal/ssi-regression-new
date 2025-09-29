"""OpenAI Responses API client + tool-calling loop."""
import json
from typing import List, Dict, Any
from openai import OpenAI

from .config import OPENAI_API_KEY, MODEL_NAME
# Tools registry maps tool name -> callable and json schema
from .tools import (
    get_growth_scores, get_current_view_state, get_indicator_scores, prioritize_focus,
    file_search, web_search
)

def _tool_spec():
    # Define function tools exposed to the model
    return [
        {
            "type": "function",
            "function": {
                "name": "get_growth_scores",
                "description": "Return final score and display domain scores for a country.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "country": {"type":"string"},
                        "year": {"type":["integer","null"]},
                        "domains": {"type":["array","null"], "items":{"type":"string"}}
                    },
                    "required":["country"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_view_state",
                "description": "Return current UI filters (country, domains, year) for a session.",
                "parameters": {
                    "type": "object",
                    "properties": {"session_id": {"type":"string"}},
                    "required":["session_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_indicator_scores",
                "description": "Return normalized indicator scores for a country.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "country": {"type":"string"},
                        "indicators": {"type":["array","null"], "items":{"type":"string"}}
                    },
                    "required":["country"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "prioritize_focus",
                "description": "Rank indicators to focus on using blended weights (no raw weights exposed).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "country": {"type":"string"},
                        "indicators": {"type":"array", "items":{"type":"string"}}
                    },
                    "required":["country","indicators"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "file_search",
                "description": "Retrieve explanatory snippets from internal docs (no numbers).",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type":"string"}},
                    "required":["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Retrieve external facts when user explicitly asks. Returns citations.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type":"string"}},
                    "required":["query"]
                }
            }
        }
    ]

TOOL_IMPL = {
    "get_growth_scores": get_growth_scores,
    "get_current_view_state": get_current_view_state,
    "get_indicator_scores": get_indicator_scores,
    "prioritize_focus": prioritize_focus,
    "file_search": file_search,
    "web_search": web_search,
}

def run_responses_loop(messages: List[Dict[str,str]], system_instructions: str, response_format_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal loop:
    - Call Responses API with tools and json_schema response_format
    - If it returns tool calls, execute and append results
    - Return final structured JSON output
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    tools = _tool_spec()

    # First call
    response = client.responses.create(
        model=MODEL_NAME,
        instructions=system_instructions,
        input=messages,
        tools=tools,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Answer",
                "schema": response_format_schema,
                "strict": True
            }
        }
    )

    # Handle tool calls via submit_tool_outputs; fall back to output_text if no structured payload is returned.
    try:
        while True:
            tool_calls = []
            if hasattr(response, "output") and response.output:
                tool_calls = [item for item in response.output if getattr(item, "type", None) == "tool_call"]

            if tool_calls:
                tool_outputs = []
                for call in tool_calls:
                    name = getattr(call, "name", None)
                    raw_args = getattr(call, "arguments", {})
                    try:
                        args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
                    except (TypeError, json.JSONDecodeError):
                        args = {}

                    impl = TOOL_IMPL.get(name)
                    if not impl:
                        result = {"error": {"code": "UNKNOWN_TOOL", "message": f"No implementation for '{name}'"}}
                    else:
                        try:
                            result = impl(**args)
                        except Exception as tool_error:
                            result = {
                                "error": {
                                    "code": "TOOL_EXECUTION_ERROR",
                                    "message": str(tool_error),
                                }
                            }

                    tool_outputs.append({
                        "tool_call_id": getattr(call, "id", None),
                        "output": json.dumps(result, ensure_ascii=False),
                    })

                response = client.responses.submit_tool_outputs(
                    response_id=response.id,
                    tool_outputs=tool_outputs,
                )
                continue

            if hasattr(response, "output") and response.output:
                for item in response.output:
                    if getattr(item, "type", None) == "message":
                        try:
                            content = getattr(item, "content", []) or []
                            first = content[0] if content else None
                            text = None
                            if first is not None:
                                text = getattr(first, "text", None)
                                if text is None and isinstance(first, dict):
                                    text = first.get("text")
                            if text:
                                return json.loads(text)
                        except Exception:
                            pass

            if hasattr(response, "output_text") and response.output_text:
                return {"kind": "explanation", "narrative": response.output_text, "citations": []}

            break
    except Exception as e:
        return {
            "kind": "explanation",
            "narrative": f"Handler error: {e}",
            "citations": [],
            "errors": [{"code": "HANDLER_ERROR", "message": str(e)}],
        }

    return {"kind": "explanation", "narrative": "No structured output returned.", "citations": []}
