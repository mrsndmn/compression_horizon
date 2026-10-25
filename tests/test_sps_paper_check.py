"""The separate manuscript gate must reject broken standalone source bundles."""

import importlib.util
from pathlib import Path


def load_checker():
    path = Path(__file__).resolve().parents[1] / "sps_paper" / "check.py"
    spec = importlib.util.spec_from_file_location("sps_article_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checker_detects_missing_local_inputs_and_figures(tmp_path):
    checker = load_checker()
    (tmp_path / "main.tex").write_text(r"\input{appendix}" + "\n" + r"\includegraphics{figures/missing}" + "\n")
    errors = checker.check(tmp_path)
    assert any("appendix" in e and "missing" in e.lower() for e in errors)
    assert any("figures/missing" in e for e in errors)


def test_checker_rejects_raw_body_tables_and_nonfinite_generated_values(tmp_path):
    checker = load_checker()
    (tmp_path / "tables").mkdir()
    (tmp_path / "main.tex").write_text(r"\begin{tabular}{lr} x & 1 \\ \end{tabular}")
    (tmp_path / "tables/results.tex").write_text(r"x & NaN \\")
    errors = checker.check(tmp_path)
    assert any("raw" in e.lower() and "main.tex" in e for e in errors)
    assert any("nonfinite" in e.lower() and "results.tex" in e for e in errors)


def test_checker_accepts_resolved_local_assets_and_rejects_unknown_citation(tmp_path):
    checker = load_checker()
    (tmp_path / "tables").mkdir()
    (tmp_path / "figures").mkdir()
    (tmp_path / "main.tex").write_text(
        r"\input{tables/results}\includegraphics{figures/curve}\citep{supported}\bibliography{references}"
    )
    (tmp_path / "tables/results.tex").write_text(r"\begin{tabular}{lr}x & 2\\\end{tabular}")
    (tmp_path / "figures/curve.pdf").write_bytes(b"figure placeholder")
    (tmp_path / "references.bib").write_text("@article{supported, title={Measured result}, year={2026}}")
    assert checker.check(tmp_path) == []
    with (tmp_path / "main.tex").open("a") as handle:
        handle.write(r"\cite{unsupported}")
    assert any("unsupported" in e for e in checker.check(tmp_path))
