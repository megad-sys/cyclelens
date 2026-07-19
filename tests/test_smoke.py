import importlib


def test_package_imports():
    module = importlib.import_module("src")
    assert module is not None
