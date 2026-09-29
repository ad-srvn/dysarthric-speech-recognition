.DEFAULT_GOAL := help

.PHONY: help start all matrix matrix-resume matrix-10 matrix-10-resume matrix-plots repeated-plots prepare test htk-check check clean-generated

VENV_PYTHON := .venv/bin/python3
PYTHON ?= $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),python3)
CONFIG ?= config/experiment.json

help:
	@echo "make start             Choose speaker group and run count"
	@echo "make all               Run one configured experiment"
	@echo "make matrix            Run the default dysarthria/model matrix"
	@echo "make matrix-resume     Resume the one-pass matrix"
	@echo "make matrix-10         Run and average ten complete matrices"
	@echo "make matrix-10-resume  Resume the ten-repeat matrix"
	@echo "make prepare           Prepare splits and MFCC features"
	@echo "make check             Check HTK and run unit tests"
	@echo "make clean-generated   Remove derived data, models, and results"

start:
	$(PYTHON) run.py

all:
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG)

matrix:
	$(PYTHON) scripts/run_experiment_matrix.py --config $(CONFIG)

matrix-resume:
	$(PYTHON) scripts/run_experiment_matrix.py --config $(CONFIG) --skip-completed

matrix-10:
	$(PYTHON) scripts/run_repeated_experiment_matrix.py --config $(CONFIG) --repeats 10

matrix-10-resume:
	$(PYTHON) scripts/run_repeated_experiment_matrix.py --config $(CONFIG) --repeats 10 --skip-completed

matrix-plots:
	$(PYTHON) scripts/plot_experiment_matrix.py

repeated-plots:
	$(PYTHON) scripts/plot_repeated_experiment_matrix.py

prepare:
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --stop-after features

test:
	$(PYTHON) -m unittest discover -s tests -v

htk-check:
	$(PYTHON) scripts/check_htk.py

check: htk-check test

clean-generated:
	$(PYTHON) scripts/clean_generated.py
