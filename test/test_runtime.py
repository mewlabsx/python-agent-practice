from runtime import execute_tool_call
from tools import tools


def test_calculator_success():
    result = execute_tool_call(
        {
            "tool": "calculator",
            "arguments": {
                "a": 10,
                "b": 2,
                "operation": "divide"
            }
        },
        tools
    )

    assert result["success"] is True
    assert result["result"] == 5


def test_calculator_zero_division():
    result = execute_tool_call(
        {
            "tool": "calculator",
            "arguments": {
                "a": 10,
                "b": 0,
                "operation": "divide"
            }
        },
        tools
    )

    assert result["success"] is False
    assert "ZeroDivisionError" in result["error"]


def test_missing_arguments():
    result = execute_tool_call(
        {
            "tool": "calculator",
            "arguments": {
                "a": 10
            }
        },
        tools
    )

    assert result["success"] is False
    assert "TypeError" in result["error"]


def test_tool_not_found():
    result = execute_tool_call(
        {
            "tool": "unknown_tool",
            "arguments": {}
        },
        tools
    )

    assert result["success"] is False
    assert "not found" in result["error"]


def test_project_not_found():
    result = execute_tool_call(
        {
            "tool": "get_project_info",
            "arguments": {
                "project_name": "UNKNOWN"
            }
        },
        tools
    )

    assert result["success"] is False
    assert "ValueError" in result["error"]
