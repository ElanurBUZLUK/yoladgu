import pytest

def test_basic():
    """Basit bir test"""
    assert 1 + 1 == 2

def test_string():
    """String testi"""
    assert "hello" + " world" == "hello world"

def test_list():
    """List testi"""
    test_list = [1, 2, 3]
    assert len(test_list) == 3
    assert test_list[0] == 1 