from pathlib import Path
from agent import run_agent
from tools import tools, openai_tools, create_retrieve_knowledge_tool
from memory import write_memory, retrieve_memory, memory_store
from runtime import execute_tool_call
from ingestion import load_documents, chunk_documents, build_index
from knowledge_schema import indicator_catalog


state = {
    "messages": [],
    "turn_count": 0,
    "model_call_count": 0,
    "tool_call_count": 0
}


documents = load_documents(Path(__file__).resolve().parent / "knowledge")
knowledge_chunks = chunk_documents(documents, indicator_catalog)
knowledge_index = build_index(knowledge_chunks)

knowledge_base = {
    "chunks": knowledge_chunks,
    "index": knowledge_index
}

# 用闭包绑定了 knowledge_base 的版本，覆盖掉 tools.py 里那个依赖模块级变量的同名函数
retrieve_knowledge = create_retrieve_knowledge_tool(knowledge_base)
tools["retrieve_knowledge"] = retrieve_knowledge


if __name__ == "__main__":
    answer1 = run_agent(
        "怎么验证掉电恢复能力？",
        state,
        tools,
        openai_tools,
        knowledge_base
    )
    print("Turn 1:", answer1)

    answer2 = run_agent(
        "计算123乘以4567",
        state,
        tools,
        openai_tools,
        knowledge_base
    )
    print("Turn 2:", answer2)

    answer3 = run_agent(
        "你好",
        state,
        tools,
        openai_tools,
        knowledge_base
    )
    print("Turn 3:", answer3)
