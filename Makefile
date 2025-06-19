
.PHONY: install
install:
	uv sync

.PHONY: train
train:
	uv run src/main.py

.PHONY: clean
clean:
	@echo "Cleaning up project artifacts.."
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete
	rm -rf build dist src/*.egg-info

