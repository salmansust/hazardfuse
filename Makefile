# make install | make test | make smoke | make train | make ablation | make check
PY ?= python
CONFIG ?= configs/glen_lyn.yaml

install:
	$(PY) -m pip install -r requirements.txt -e .

test:
	$(PY) -m pytest -q

smoke:
	$(PY) scripts/make_synthetic_data.py
	$(PY) scripts/train.py --config configs/smoke.yaml

train:
	$(PY) scripts/train.py --config $(CONFIG)

ablation:
	$(PY) scripts/ablation.py --config $(CONFIG)

check:
	$(PY) scripts/check_reproduction.py --config $(CONFIG)

.PHONY: install test smoke train ablation check
