"""
agent_trace.py —— 离线打印 Agent Loop 的完整执行轨迹

用途
    1. 不调用真实模型、不需要 API Key、不花钱，就能看清 run_agent 的内部循环
    2. 面试时可以直接演示「一轮对话在代码里到底发生了什么」

原理
    把 provider.call_model 替换成一个「假模型」，返回预设好的响应。
    其余代码（agent.py / runtime.py / shape.py / memory.py）全部原样执行。
    这本身也是一种工程能力：模块边界清晰，才能被替换掉。

用法
    python agent_trace.py

关于 knowledge_base
    run_agent 的签名要求传 knowledge_base，但函数体内从未使用它（死参数）。
    本脚本显式传 None 占位。若日后删掉该参数，这里两处调用也要一起删。
"""

import agent
import memory
from agent import run_agent
from memory import memory_store
from runtime import success_result
from tools import tools, openai_tools


# ============================================================
# 打印辅助
# ============================================================

def line(text):
    print(text)


def rule(text):
    print()
    print("=" * 68)
    print(text)
    print("=" * 68)


def describe(item, index):
    """把一条消息压缩成一行，方便看消息数组的形状"""
    role = str(item.get("role"))
    content = item.get("content")
    tool_calls = item.get("tool_calls")

    prefix = "   #" + str(index) + " " + role.ljust(10)

    if tool_calls:
        names = []
        for tc in tool_calls:
            function = tc.get("function", {})
            names.append(str(function.get("name")))
        return prefix + "tool_calls = " + str(names)

    if content is None:
        return prefix + "(content 为空)"

    text = content.replace("\n", " ")
    if len(text) > 44:
        text = text[:44] + "..."
    return prefix + 'content = "' + text + '"'


def dump_messages(messages):
    for index in range(len(messages)):
        line(describe(messages[index], index))


# ============================================================
# 假模型：按剧本返回
# ============================================================

def make_fake_model(script, label):
    """script 是预设的响应列表，每次调用取一条。"""

    box = {"index": 0}

    def fake_call_model(messages, tools_arg=None, max_retries=2, retry_delay=1):

        line("")
        line("   [模型被调用第 " + str(box["index"] + 1) + " 次] 它实际看到的上下文：")
        dump_messages(messages)

        if box["index"] < len(script):
            message = script[box["index"]]
        else:
            message = {"role": "assistant", "content": "(" + label + " 剧本已用完)"}

        box["index"] = box["index"] + 1

        return success_result(message)

    return fake_call_model


def build_state():
    return {
        "messages": [],
        "turn_count": 0,
        "model_call_count": 0,
        "tool_call_count": 0,
    }


# ============================================================
# 场景 1：正常一轮 —— 先要工具，再给答案
# ============================================================

rule("场景 1  正常执行：模型先调工具，拿到结果后再回答")

call_answer = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_001",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": '{"a": 123, "b": 456, "operation": "multiply"}',
            },
        }
    ],
}

final_answer = {
    "role": "assistant",
    "content": "123 x 456 = 56088",
}

agent.call_model = make_fake_model([call_answer, final_answer], "场景1")

# memory 模块内部也会调模型（用于抽取记忆）。
# 这里让它永远返回空列表，避免真的发网络请求。
memory.call_model = lambda messages, tools_arg=None, max_retries=2, retry_delay=1: success_result(
    {"role": "assistant", "content": "[]"}
)

state = build_state()

line("")
line('调用 run_agent("帮我算 123 x 456")')
line("")
line("   state['messages'] 进入时是空的，run_agent 先追加一条 user 消息：")

answer = run_agent("帮我算 123 x 456", state, tools, openai_tools, knowledge_base=None)

line("")
line("   最终 state['messages'] 里有 " + str(len(state["messages"])) + " 条：")
dump_messages(state["messages"])

line("")
line("   返回给调用方的是：" + repr(answer))
line("")
line("   state 计数器：模型调用 " + str(state["model_call_count"]) + " 次，工具调用 " + str(state["tool_call_count"]) + " 次")


# ============================================================
# 场景 2：模型一直要工具 —— 撞上 max_steps
# ============================================================

rule("场景 2  异常路径：模型一直调工具，撞上 max_steps 会怎样？")

endless = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_loop",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": '{"a": 1, "b": 1, "operation": "add"}',
            },
        }
    ],
}

agent.call_model = make_fake_model([endless] * 10, "场景2")

state2 = build_state()

line("")
line('调用 run_agent("循环吧", max_steps=3)')
line("")

answer2 = run_agent("循环吧", state2, tools, openai_tools, knowledge_base=None, max_steps=3)

line("")
line("   state['messages'] 里有 " + str(len(state2["messages"])) + " 条（注意结尾没有 assistant 回复）")
line("   返回给调用方的是：" + repr(answer2))
line("")
line("   >>> 问题在这里：run_agent 的 while 循环走完后没有 return 语句，")
line("   >>> 函数隐式返回 None。调用方拿到 None，无法区分「执行失败」和「步数用尽」。")
line("   >>> 而且 state['messages'] 末尾停在 tool 消息上，这一轮是残缺的，")
line("   >>> 下一轮再进来，模型看到的历史会以 tool 消息结尾。")


# ============================================================
# 记忆部分
# ============================================================

rule("附：memory_store 当前内容（进程内字典，脚本结束即丢）")

memory.write_memory("preference", "report_language", "Chinese")
line("   写入 write_memory('preference', 'report_language', 'Chinese') 后：")
line("   " + str(memory_store))
line("")
line("   注意 memory_store 是普通 dict，没有持久化。")
line("   README 里已标注这是 MVP 限制，面试时主动说出来比被问出来好。")
