import pytest

from rag import (
    analyze_query,
    rewrite_query,
    filter_candidates,
    cosine_similarity,
    rank_by_similarity,
    retrieve
)
from ingestion import (
    build_embedding_text,
    build_index
)


test_chunks = [
    {
        "chunk_id": "IND-023.argumentation.001",
        "source": "test.md",
        "chapter_title": "高可用能力论证",
        "chapter_role": "argumentation",
        "indicator_id": "IND-023",
        "indicator_name": "断电自恢复",
        "indicator_aliases": [
            "断电自恢复",
            "掉电恢复",
            "断电恢复",
            "供电恢复后自动恢复"
        ],
        "part_index": 1,
        "text": "系统需要具备断电后的自动恢复能力。"
    },
    {
        "chunk_id": "IND-023.design.001",
        "source": "test.md",
        "chapter_title": "故障恢复设计",
        "chapter_role": "design",
        "indicator_id": "IND-023",
        "indicator_name": "断电自恢复",
        "indicator_aliases": [
            "断电自恢复",
            "掉电恢复",
            "断电恢复",
            "供电恢复后自动恢复"
        ],
        "part_index": 1,
        "text": "供电恢复后自动启动核心组件。"
    },
    {
        "chunk_id": "IND-023.test.001",
        "source": "test.md",
        "chapter_title": "故障恢复测试",
        "chapter_role": "test",
        "indicator_id": "IND-023",
        "indicator_name": "断电自恢复",
        "indicator_aliases": [
            "断电自恢复",
            "掉电恢复",
            "断电恢复",
            "供电恢复后自动恢复"
        ],
        "part_index": 1,
        "text": "恢复供电后检查业务服务是否自动恢复。"
    }
]


# ---- Query Understanding ----

def test_indicator_name_resolution():
    result = analyze_query("断电自恢复怎么实现？")

    assert result["indicator_id"] == "IND-023"


def test_indicator_alias_resolution():
    result = analyze_query("怎么验证掉电恢复能力？")

    assert result["indicator_id"] == "IND-023"


def test_test_role_resolution():
    result = analyze_query("怎么验证掉电恢复能力？")

    assert result["chapter_role"] == "test"


# ---- Metadata Filter ----

def test_metadata_filter():
    query_plan = analyze_query("怎么验证掉电恢复能力？")
    result = filter_candidates(test_chunks, query_plan)

    assert len(result) == 1
    assert result[0]["chunk_id"] == "IND-023.test.001"


def test_unknown_query_does_not_remove_all_chunks(monkeypatch):
    # Explicitly simulate a model that cannot resolve the query.
    # This test checks filtering behavior, not the live provider.
    monkeypatch.setattr("rag.call_model", lambda messages: {
        "success": True, "result": {"content": "{}"}, "error": None
    })
    query_plan = analyze_query("系统异常情况下有哪些恢复能力？")
    result = filter_candidates(test_chunks, query_plan)

    assert len(result) == len(test_chunks)


# ---- Embedding Text ----

def test_build_embedding_text():
    text = build_embedding_text(test_chunks[0])

    assert "断电自恢复" in text
    assert "argumentation" in text
    assert "高可用能力论证" in text
    assert test_chunks[0]["text"] in text


# ---- Cosine Similarity ----

def test_cosine_same_direction():
    result = cosine_similarity([1, 1], [2, 2])

    assert abs(result - 1.0) < 0.0001


def test_cosine_orthogonal():
    result = cosine_similarity([1, 0], [0, 1])

    assert abs(result) < 0.0001


def test_cosine_zero_vector():
    result = cosine_similarity([0, 0], [1, 1])

    assert result == 0.0


def test_cosine_dimension_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])


# ---- Similarity Ranking ----

def test_similarity_ranking():
    candidates = [
        {"chunk_id": "A"},
        {"chunk_id": "B"}
    ]
    embeddings = {
        "A": [1, 0],
        "B": [0, 1]
    }

    result = rank_by_similarity([0.9, 0.1], candidates, embeddings, top_k=1)

    assert result[0]["chunk"]["chunk_id"] == "A"


def test_retrieve_pipeline():
    index = build_index(test_chunks)
    result = retrieve("怎么验证掉电恢复能力？", test_chunks, index, top_k=2)

    assert result["query_plan"]["indicator_id"] == "IND-023"
    assert result["query_plan"]["chapter_role"] == "test"
    assert len(result["results"]) >= 1
    assert result["results"][0]["chunk"]["chunk_id"] == "IND-023.test.001"


def test_query_rewrite_argumentation():
    result = analyze_query("为什么需要掉电恢复能力？")

    assert result["semantic_query"] == "断电自恢复；必要性、需求背景和论证依据"


def test_query_rewrite_design():
    result = analyze_query("掉电恢复是怎么实现的？")

    assert "断电自恢复" in result["semantic_query"]
    assert "实现机制" in result["semantic_query"]


def test_query_rewrite_fallback():
    query = "系统异常以后会怎么样？"
    result = rewrite_query(query, indicator_id=None, chapter_role=None)

    assert result == query
