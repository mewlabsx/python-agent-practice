"""形状转换层 —— 全项目唯一的「文本 ⇄ 结构」边界出口。

为什么需要这个文件：
    数据在本项目里有三种形状：
        A. SDK 对象（pydantic 模型）—— 点号取值，如 message.tool_calls[0].function.name
        B. Python dict / list      —— .get / [] / for 取值，业务逻辑的主场
        C. JSON 字符串             —— 跨进程传输时的文本形态，不能取值、不能遍历

    形状只在边界处切换，边界只有两处：
        - 与模型服务通信（跨进程，只能传文本）
        - 与 SDK 交互（库返回的是对象，不是 dict）

    把转换收拢在这里，是为了让"边界在哪"一眼可见：
    全项目搜索 json.loads / json.dumps，应该只在本文件里出现。
    业务代码（agent.py / memory.py）只处理形状 B，不再关心文本与对象。

设计原则：
    1. 只做转换，不做校验、不做业务判断
    2. 解析失败不抛异常，返回默认值 —— 模型输出不是合法 JSON 是常态，不是异常
    3. 调用方不需要知道拿到的是 A 还是 B（to_dict 自动判断）
"""

import json


def to_dict(obj):
    """形状 A / B → B

    SDK 对象走 model_dump，dict 原样返回。
    统一入口，调用方不用写 hasattr 试探。

    形状契约
        入参：SDK 对象 或 dict
        返回：dict
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    return obj


def to_json(data):
    """形状 B → C

    把 dict / list 变成文本，准备跨进程发出去。
    ensure_ascii=False 是为了让中文保持原样，便于阅读和排查。

    形状契约
        入参：dict 或 list
        返回：str
    """
    return json.dumps(data, ensure_ascii=False)


def from_json(text, default=None):
    """形状 C → B

    把文本解析成 dict / list。
    非字符串、或解析失败时返回 default，不抛异常。

    形状契约
        入参：str（或任何东西 —— 非 str 直接返回 default）
        返回：dict / list，或 default
    """
    if not isinstance(text, str):
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default
