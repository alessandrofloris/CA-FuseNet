PYTHON ?= python

.PHONY: install dev train lint format test clean

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

train_pose:
	$(PYTHON) -m ca_fusenet.scripts.train experiment=pose_baseline hydra.job.name=train_pose

train_video:
	$(PYTHON) -m ca_fusenet.scripts.train experiment=video_baseline hydra.job.name=train_video
	
train_cafusenet:
	$(PYTHON) -m ca_fusenet.scripts.train_ca_fusenet experiment=ca_fusenet hydra.job.name=train_cafusenet

eval_pose:
	$(PYTHON) -m ca_fusenet.scripts.eval experiment=pose_baseline hydra.job.name=eval_pose

eval_video:
	$(PYTHON) -m ca_fusenet.scripts.eval experiment=video_baseline hydra.job.name=eval_video

analyze_results:
	$(PYTHON) -m ca_fusenet.scripts.analyze_results --experiment_dirs outputs/2026-03-06/11-15-15_video_baseline/artifacts/reports/ --output_dir analysis_output --top_k 12

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
