PYTHON ?= python

.PHONY: install dev train lint format test clean

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

train_pose:
	$(PYTHON) -m ca_fusenet.scripts.train model=pose_stgcn_baseline training=pose

train_video:
	$(PYTHON) -m ca_fusenet.scripts.train model=rgb_r3d18_baseline training=video

eval_pose:
	$(PYTHON) -m ca_fusenet.scripts.eval

extract_tublets:
	$(PYTHON) -m ca_fusenet.scripts.extract_tublets

lint:
	ruff check src

format:
	black src

test:
	pytest -q

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['build','dist','.pytest_cache','outputs']]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').glob('*.egg-info')]"
