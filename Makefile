PYTHON ?= /home/jovyan/.mlspace/envs/compression_horizon/bin/python
LATEXMK ?= /home/jovyan/.TinyTeX/bin/x86_64-linux/latexmk

.PHONY: paper paper-check paper-clean tables sps-paper sps-paper-check sps-paper-figures sps-paper-clean

paper: paper-check
	cd paper && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error example_paper.tex
	@echo "PDF written to paper/build/example_paper.pdf"

paper-check:
	$(PYTHON) paper/lint_paper.py

paper-clean:
	cd paper && $(LATEXMK) -C
	rm -rf paper/build

# Re-render every paper table from artifacts. Not a dependency of `paper`
# because attention-mass and aggregation passes are slow; invoke explicitly
# after the relevant experiments finish.
tables:
	PY=$(PYTHON) bash scripts/paper/tables/tables.sh

# The SPS article has its own evidence snapshots and build; the original paper
# and its canonical cohort checks are independent.
sps-paper:
	$(MAKE) -C sps_paper pdf PYTHON="$(PYTHON)" LATEXMK="$(LATEXMK)"

sps-paper-check:
	$(MAKE) -C sps_paper check PYTHON="$(PYTHON)"

sps-paper-figures:
	$(MAKE) -C sps_paper assets PYTHON="$(PYTHON)"

sps-paper-clean:
	$(MAKE) -C sps_paper clean LATEXMK="$(LATEXMK)"
