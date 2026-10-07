from shape import from_json, to_json
from provider import call_model
from runtime import success_result, failure_result


# ---- 记忆存储 ----
# 纯进程内字典，脚本结束即丢。README 里已标注这是 MVP 限制。
memory_store = {
    "preference": {},
    "semantic": {}
}


# ---- 记忆读写 ----
def write_memory(memory_type, key, value):
    if memory_type not in memory_store:
        raise ValueError(f"Unknown memory type: {memory_type}")

    memory_store[memory_type][key] = value


def retrieve_memory(memory_type, key):
    if memory_type in memory_store:
        return memory_store[memory_type].get(key, None)
    return None


# ---- 记忆提取策略 ----
MEMORY_EXTRACTOR_PROMPT = """
You are a memory extraction component.

Analyze one completed agent turn and extract information
that is useful across future sessions.

Allowed memory types:
- preference: stable user preferences
- semantic: relatively stable factual information

Do not save:
- one-time requests
- temporary runtime events
- tool errors
- ordinary calculations
- unsupported assumptions

Return JSON only.

Output format:
[
    {
        "memory_type": "preference or semantic",
        "key": "stable_key",
        "value": "memory_value"
    }
]

If there is nothing worth saving, return:
[]
"""


def validate_memory_candidate(candidate):
    if not isinstance(candidate, dict):
        return failure_result("Invalid memory candidate format")

    memory_type = candidate.get("memory_type")
    key = candidate.get("key")
    value = candidate.get("value")

    if memory_type not in ("preference", "semantic"):
        return failure_result("Invalid memory type")

    if not isinstance(key, str) or not key.strip():
        return failure_result("Invalid memory key")

    if value is None:
        return failure_result("Invalid memory value")

    return success_result("validate_memory_candidate success")


# 送模型之前先剥掉消息里的冗余字段，只留三类消息各自需要的键
def sanitize_turn_for_memory(current_turn):
    sanitized_turn = []

    for item in current_turn:
        role = item.get("role")

        if role == "user":
            sanitized_turn.append({
                "role": "user",
                "content": item.get("content")
            })

        elif role == "assistant":
            sanitized_item = {
                "role": "assistant",
                "content": item.get("content")
            }

            tool_calls = []
            for tool_call in item.get("tool_calls", []):
                function = tool_call.get("function", {})
                sanitized_tool_call = {
                    "id": tool_call.get("id"),
                    "type": tool_call.get("type"),
                    "function": {
                        "name": function.get("name"),
                        "arguments": function.get("arguments")
                    }
                }
                tool_calls.append(sanitized_tool_call)

            if tool_calls:
                sanitized_item["tool_calls"] = tool_calls

            sanitized_turn.append(sanitized_item)

        elif role == "tool":
            sanitized_turn.append({
                "role": "tool",
                "tool_call_id": item.get("tool_call_id"),
                "content": item.get("content")
            })

    return sanitized_turn


def evaluate_memory_write(current_turn):
    sanitized_turn = sanitize_turn_for_memory(current_turn)
    current_turn_json = to_json(sanitized_turn)

    extractor_messages = [
        {
            "role": "system",
            "content": MEMORY_EXTRACTOR_PROMPT
        },
        {
            "role": "user",
            "content": current_turn_json
        }
    ]

    model_result = call_model(extractor_messages)
    if not model_result["success"]:
        return []

    message = model_result["result"]
    content = message.get("content")
    raw_candidates = from_json(content)

    if not isinstance(raw_candidates, list):
        return []

    validated_candidates = []
    for candidate in raw_candidates:
        validation_result = validate_memory_candidate(candidate)
        if validation_result["success"]:
            validated_candidates.append(candidate)

    return validated_candidates
