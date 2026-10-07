from shape import from_json, to_json
from provider import call_model
from runtime import execute_tool_call, failure_result
from memory import retrieve_memory, evaluate_memory_write, write_memory
from rag import build_rag_context


# 只保留最近 max_turns 轮用户消息开始的上下文，前面的一律丢掉；
# 记忆若存在，作为一条 system 消息插在最前面
def build_model_context(messages, retrieve_memory=None, max_turns=2):
    user_indexes = []
    for index, item in enumerate(messages):
        if item.get("role") == "user":
            user_indexes.append(index)

    if len(user_indexes) <= max_turns:
        model_context = list(messages)
    else:
        start_index = user_indexes[-max_turns]
        model_context = list(messages[start_index:])

    if retrieve_memory:
        model_context.insert(0, {
            "role": "system",
            "content": "Relevant memory: " + retrieve_memory
        })

    return model_context


def run_agent(user_input, state, tools, openai_tools, knowledge_base, max_steps=5):
    turn_start_index = len(state["messages"])

    state["turn_count"] += 1
    state["messages"].append({
        "role": "user",
        "content": user_input
    })

    steps = 0

    while steps < max_steps:
        # 每轮都重新读一次记忆：上一轮可能刚写进去新的偏好
        memory_report_language = retrieve_memory("preference", "report_language")
        if memory_report_language is not None:
            retrieve_memory_content = "The user's preferred report language is " + memory_report_language
        else:
            retrieve_memory_content = None

        model_context = build_model_context(state["messages"], retrieve_memory_content, max_turns=2)
        model_result = call_model(model_context, openai_tools)

        if not model_result["success"]:
            return model_result["error"]

        state["model_call_count"] += 1
        message = model_result["result"]
        steps = steps + 1

        tool_calls = message.get("tool_calls", [])

        # 模型不再要工具了 -> 这就是最终回答，顺便结算这一轮该不该写记忆
        if not tool_calls:
            state["messages"].append(message)

            current_turn = state["messages"][turn_start_index:]
            memory_candidates = evaluate_memory_write(current_turn)

            for candidate in memory_candidates:
                write_memory(
                    candidate["memory_type"],
                    candidate["key"],
                    candidate["value"]
                )

            return message.get("content")

        state["messages"].append(message)

        for api_tool_call in tool_calls:
            state["tool_call_count"] += 1

            function = api_tool_call.get("function", {})
            tool_name = function.get("name")
            arguments_json = function.get("arguments")

            arguments = from_json(arguments_json)

            # 模型给的参数不是合法 JSON 对象是常态，不抛异常，包成失败结果回传
            if not isinstance(arguments, dict):
                execution_result = failure_result("Error: Tool arguments must be a valid JSON object")
            else:
                tool_call = {
                    "tool": tool_name,
                    "arguments": arguments
                }
                execution_result = execute_tool_call(tool_call, tools)

            state["messages"].append({
                "role": "tool",
                "tool_call_id": api_tool_call.get("id"),
                "content": to_json(execution_result)
            })
