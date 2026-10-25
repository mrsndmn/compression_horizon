PYTHON ?= python
LATEXMK ?= latexmk

.PHONY: all pdf assets check clean

all: pdf

assets:
	$(PYTHON) reproduce.py render

check:
	$(PYTHON) check.py

pdf: assets
	$(PYTHON) check.py
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(LATEXMK) -C main.tex
