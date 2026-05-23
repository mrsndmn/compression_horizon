PYTHON ?= /home/jovyan/.mlspace/envs/compression_horizon/bin/python
LATEXMK ?= /home/jovyan/.TinyTeX/bin/x86_64-linux/latexmk

.PHONY: paper paper-check paper-clean

paper: paper-check
	cd paper && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error example_paper.tex

paper-check:
	$(PYTHON) paper/check_unused_attachments.py

paper-clean:
	cd paper && $(LATEXMK) -C
