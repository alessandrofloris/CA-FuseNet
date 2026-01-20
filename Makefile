PYTHON ?= python

.PHONY: install dev train lint format test clean

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

train:
	$(PYTHON) -m ca_fusenet.scripts.train

train_pose:
	$(PYTHON) -m ca_fusenet.training.train_pose_baseline

lint:
	ruff check src

format:
	black src

test:
	pytest -q

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['build','dist','.pytest_cache','outputs']]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').glob('*.egg-info')]"
