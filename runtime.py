# 统一输出：成功
def success_result(result):
    return {
        "success": True,
        "result": result,
        "error": None
    }


# 统一输出：失败
def failure_result(error):
    return {
        "success": False,
        "result": None,
        "error": error
    }


# 工具执行器
def execute_tool_call(tool_call, tools):
    if not isinstance(tool_call, dict):
        return failure_result("Error: Tool call must be a dictionary")

    tool_name = tool_call.get("tool")
    if tool_name is None:
        return failure_result("Error: Missing tool name")

    arguments = tool_call.get("arguments", {})
    if not isinstance(arguments, dict):
        return failure_result("Error: Arguments must be a dictionary")

    tool = tools.get(tool_name)
    if tool is None:
        return failure_result(f"Error: Tool '{tool_name}' not found")

    try:
        print(f"Executing tool: {tool_name}")
        print("Arguments:", arguments)

        result = tool(**arguments)
        return success_result(result)

    except Exception as err:
        return failure_result(f"Error: {type(err).__name__}: {str(err)}")
