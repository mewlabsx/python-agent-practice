import math

from embedding_provider import embed_text
from ingestion import load_documents, chunk_documents, build_index
from knowledge_schema import indicator_catalog
from provider import call_model
from shape import from_json


chapter_role_keywords = {
    "argumentation": ["为什么", "必要性", "需求", "论证"],
    "design": ["怎么实现", "如何实现", "怎么处理", "实现", "设计", "机制", "原理"],
    "test": ["怎么测", "如何测试", "测试", "验证", "判据"]
}

role_descriptions = {
    "argumentation": "必要性、需求背景和论证依据",
    "design": "设计方案、实现机制和处理流程",
    "test": "测试方法、验证步骤和判定标准"
}

QUERY_UNDERSTANDING_PROMPT = """
You are a query understanding component for a technical knowledge base.

Your task is to analyze a user query and infer missing retrieval fields.

Allowed chapter roles:
- argumentation: requirement background, necessity, justification
- design: implementation, mechanism, architecture, processing flow
- test: test method, verification, acceptance criteria

You must only use indicator IDs that exist in the provided indicator catalog.

If you cannot determine a field reliably, return null.

Do not answer the user's question.
Do not invent technical facts.

Return JSON only.

Output format:

{
    "indicator_id": "known indicator id or null",
    "chapter_role": "argumentation, design, test, or null",
    "semantic_query": "a concise retrieval-oriented rewrite of the user's query"
}
"""


def resolve_indicator(query):
    for indicator_id, indicator in indicator_catalog.items():
        for alias in indicator["aliases"]:
            if alias in query:
                return {
                    "indicator_id": indicator_id,
                    "confidence": "high"
                }

    return {
        "indicator_id": None,
        "confidence": "none"
    }


def resolve_chapter_role(query):
    for role, keywords in chapter_role_keywords.items():
        for keyword in keywords:
            if keyword in query:
                return {
                    "chapter_role": role,
                    "confidence": "high"
                }

    return {
        "chapter_role": None,
        "confidence": "none"
    }


# 规则命中时把 query 规范化成「标准指标名 + 标准章节角色描述」；
# 什么都没命中就原样返回，交给后面的 LLM 兜底。
def rewrite_query(query, indicator_id=None, chapter_role=None):
    parts = []

    if indicator_id is not None:
        indicator = indicator_catalog.get(indicator_id)
        if indicator is not None:
            indicator_name = indicator.get("name")
            if indicator_name:
                parts.append(indicator_name)

    if chapter_role is not None:
        role_description = role_descriptions.get(chapter_role)
        if role_description:
            parts.append(role_description)

    if parts:
        return "；".join(parts)

    return query


def build_indicator_catalog_text():
    lines = []
    for indicator_id, indicator in indicator_catalog.items():
        aliases = "；".join(indicator["aliases"])
        lines.append(f"{indicator_id}: {indicator['name']} (aliases: {aliases})")
    return "\n".join(lines)


def analyze_query_with_llm(query, partial_plan):
    catalog_text = build_indicator_catalog_text()

    messages = [
        {
            "role": "system",
            "content": QUERY_UNDERSTANDING_PROMPT
        },
        {
            "role": "user",
            "content": (
                f"Indicator catalog:\n"
                f"{catalog_text}\n\n"
                f"User query:\n"
                f"{query}\n\n"
                f"Fields already resolved by deterministic rules:\n"
                f"{partial_plan}"
            )
        }
    ]

    model_result = call_model(messages)
    if not model_result["success"]:
        return None

    message = model_result["result"]
    content = message.get("content")
    result = from_json(content)

    if not isinstance(result, dict):
        return None

    return result


# 模型返回的东西一律不可信：指标编号必须在目录里，角色必须在白名单里，才收下
def validate_llm_query_result(result):
    if not isinstance(result, dict):
        return {}

    validated = {}

    indicator_id = result.get("indicator_id")
    if indicator_id in indicator_catalog:
        validated["indicator_id"] = indicator_id

    chapter_role = result.get("chapter_role")
    if chapter_role in ("argumentation", "design", "test"):
        validated["chapter_role"] = chapter_role

    semantic_query = result.get("semantic_query")
    if isinstance(semantic_query, str) and semantic_query.strip():
        validated["semantic_query"] = semantic_query.strip()

    return validated


# ---- Query Understanding Pipeline ----

def analyze_query(query):
    indicator_result = resolve_indicator(query)
    role_result = resolve_chapter_role(query)

    indicator_id = indicator_result["indicator_id"]
    chapter_role = role_result["chapter_role"]

    semantic_query = rewrite_query(query, indicator_id, chapter_role)

    query_plan = {
        "need_retrieval": True,
        "scope": "indicator" if indicator_id is not None else None,
        "indicator_id": indicator_id,
        "chapter_role": chapter_role,
        "semantic_query": semantic_query,
        "filter_confidence": {
            "indicator_id": indicator_result["confidence"],
            "chapter_role": role_result["confidence"]
        }
    }

    # 规则层两个字段都命中就不用调模型了
    needs_llm = query_plan["indicator_id"] is None or query_plan["chapter_role"] is None
    if not needs_llm:
        return query_plan

    llm_result = analyze_query_with_llm(query, query_plan)
    validated = validate_llm_query_result(llm_result)

    # 模型补出来的字段置信度只算 medium —— 高于 none，但不足以驱动过滤
    if query_plan["indicator_id"] is None and "indicator_id" in validated:
        query_plan["indicator_id"] = validated["indicator_id"]
        query_plan["filter_confidence"]["indicator_id"] = "medium"
        query_plan["scope"] = "indicator"

    if query_plan["chapter_role"] is None and "chapter_role" in validated:
        query_plan["chapter_role"] = validated["chapter_role"]
        query_plan["filter_confidence"]["chapter_role"] = "medium"

    if "semantic_query" in validated:
        query_plan["semantic_query"] = validated["semantic_query"]

    return query_plan


# ---- Metadata Retrieval ----

def filter_candidates(chunks, query_plan):
    candidates = list(chunks)

    indicator_id = query_plan.get("indicator_id")
    chapter_role = query_plan.get("chapter_role")
    confidence = query_plan.get("filter_confidence", {})

    # 只有 high 置信度才允许过滤：
    # 筛错了是静默故障（结果集莫名变空，且看不出原因），宁可不筛
    if indicator_id is not None and confidence.get("indicator_id") == "high":
        candidates = [chunk for chunk in candidates if chunk.get("indicator_id") == indicator_id]

    if chapter_role is not None and confidence.get("chapter_role") == "high":
        candidates = [chunk for chunk in candidates if chunk.get("chapter_role") == chapter_role]

    return candidates


# ---- Vector Similarity ----

def cosine_similarity(vector_a, vector_b):
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimension")

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ---- Similarity Ranking ----

def rank_by_similarity(query_embedding, candidates, chunk_embeddings, top_k=3):
    scored_results = []

    for chunk in candidates:
        chunk_id = chunk["chunk_id"]
        if chunk_id not in chunk_embeddings:
            raise ValueError(f"Missing embedding for chunk: {chunk_id}")

        chunk_embedding = chunk_embeddings[chunk_id]
        score = cosine_similarity(query_embedding, chunk_embedding)

        scored_results.append({
            "chunk": chunk,
            "score": score
        })

    scored_results.sort(key=lambda item: item["score"], reverse=True)
    return scored_results[:top_k]


# ---- Retrieval Pipeline ----

def retrieve(query, chunks, index, top_k=3):
    query_plan = analyze_query(query)
    candidates = filter_candidates(chunks, query_plan)

    query_embedding = embed_text(query_plan["semantic_query"])
    results = rank_by_similarity(query_embedding, candidates, index, top_k=top_k)

    return {
        "query_plan": query_plan,
        "results": results
    }


def format_retrieval_context(results):
    evidence_blocks = []

    for index, item in enumerate(results, start=1):
        chunk = item["chunk"]

        block = (
            f"[Evidence {index}]\n"
            f"Chunk ID: {chunk.get('chunk_id', '')}\n"
            f"指标：{chunk.get('indicator_name', '')}\n"
            f"章节：{chunk.get('chapter_title', '')}\n"
            f"内容：\n"
            f"{chunk.get('text', '')}"
        )

        evidence_blocks.append(block)

    return "\n\n".join(evidence_blocks)


def build_rag_context(query, chunks, index, top_k=3):
    retrieval_result = retrieve(query, chunks, index, top_k=top_k)
    context = format_retrieval_context(retrieval_result["results"])

    return {
        "query_plan": retrieval_result["query_plan"],
        "context": context
    }


# ---- Demo / Manual Test ----

if __name__ == "__main__":
    documents = load_documents("knowledge")
    chunks = chunk_documents(documents, indicator_catalog, max_chars=500)
    knowledge_index = build_index(chunks)

    test_queries = [
        "为什么需要掉电恢复能力？",
        "掉电恢复是怎么实现的？",
        "怎么验证掉电恢复能力？",
        "掉电以后系统会发生什么？"
    ]

    for query in test_queries:
        print("\n==============================")
        print("Query:", query)

        retrieval_result = retrieve(query, chunks, knowledge_index, top_k=3)

        print("Query Plan:")
        print(retrieval_result["query_plan"])

        print("\nRetrieved Chunks:")
        for item in retrieval_result["results"]:
            print(item["chunk"]["chunk_id"], item["score"])

    # 循环结束后 query 停留在最后一个元素上，这里是对它再跑一次检索
    retrieval_result = retrieve(query, chunks, knowledge_index, top_k=3)
    context = format_retrieval_context(retrieval_result["results"])

    print("\nKnowledge Context:")
    print(context)
