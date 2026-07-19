.PHONY: install test lint

install:
	pip install -r requirements.txt

test:
	pytest -q

lint:
	python -m py_compile $(shell find src tests -name '*.py')
