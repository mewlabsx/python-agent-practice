import pytest

from memory import memory_store, write_memory, retrieve_memory


def setup_function():
    memory_store.clear()
    memory_store.update({
        "preference": {},
        "semantic": {}
    })


def test_write_and_retrieve_preference():
    write_memory("preference", "report_language", "Chinese")

    result = retrieve_memory("preference", "report_language")

    assert result == "Chinese"


def test_overwrite_memory():
    write_memory("preference", "report_language", "Chinese")
    write_memory("preference", "report_language", "English")

    result = retrieve_memory("preference", "report_language")

    assert result == "English"


def test_missing_memory():
    result = retrieve_memory("preference", "unknown_key")

    assert result is None


def test_different_memory_types():
    write_memory("preference", "project", "DTM")
    write_memory("semantic", "project", "TaskOS")

    assert retrieve_memory("preference", "project") == "DTM"
    assert retrieve_memory("semantic", "project") == "TaskOS"


def test_unknown_memory_type():
    with pytest.raises(ValueError):
        write_memory("unknown", "key", "value")

    assert "unknown" not in memory_store
