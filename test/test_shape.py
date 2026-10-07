import pytest

from shape import from_json, to_json, to_dict


def test_from_json_success():
    result = from_json('{"a": 1}')

    assert result == {"a": 1}


def test_from_json_invalid():
    result = from_json("not-json")

    assert result is None


def test_from_json_default():
    result = from_json("not-json", default={})

    assert result == {}


def test_to_json_success():
    result = to_json({
        "name": "测试",
        "value": 1
    })

    assert '"name"' in result
    assert "测试" in result


def test_to_json_invalid_object():
    with pytest.raises(TypeError):
        to_json({
            "value": object()
        })
