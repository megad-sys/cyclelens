.PHONY: install test lint

install:
	pip install -r requirements.txt

test:
	DYLD_LIBRARY_PATH="$(CURDIR)/.venv/native-libs" python3 -m pytest -q

lint:
	python -m py_compile $(shell find src tests -name '*.py')
