PYTHON ?= python
LATEXMK ?= latexmk

.PHONY: all pdf assets check clean

all: pdf

assets:
	$(PYTHON) reproduce.py render
	$(PYTHON) pretrained.py render
	$(PYTHON) numerics.py

check:
	$(PYTHON) check.py
	$(PYTHON) pretrained.py verify

pdf: assets
	$(MAKE) check PYTHON="$(PYTHON)"
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(LATEXMK) -C main.tex
