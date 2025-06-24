
.PHONY: install
install:
	uv sync

.PHONY: train-vpg
train-vpg:
	uv run src/train_vpg.py

.PHONY: train-ppo
train-ppo:
	uv run src/train_ppo.py

.PHONY: clean
clean:
	@echo "Cleaning up project artifacts.."
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete
	rm -rf build dist src/*.egg-info
