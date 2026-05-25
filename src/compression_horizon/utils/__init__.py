from __future__ import annotations

from typing import Optional


def hlines_to_booktabs(latex: str) -> str:
    r"""Convert tabulate's plain ``\hline`` rules to booktabs rules.

    tabulate's ``latex``/``latex_raw`` formats frame a table with ``\hline``
    rules (top, header separator, bottom). booktabs rules carry
    ``\aboverulesep``/``\belowrulesep`` spacing and proper rule weights, which
    look cleaner: the first ``\hline`` becomes ``\toprule``, the last
    ``\bottomrule``, and any in between ``\midrule``. Output without ``\hline``
    (non-LaTeX formats) is returned unchanged.

    Requires ``\usepackage{booktabs}`` in the document preamble.
    """
    lines = latex.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.strip() == "\\hline"]
    for n, i in enumerate(idx):
        if n == 0:
            lines[i] = "\\toprule"
        elif n == len(idx) - 1:
            lines[i] = "\\bottomrule"
        else:
            lines[i] = "\\midrule"
    return "\n".join(lines)


def to_mean_std_cell(
    val_mean: Optional[float], val_std: Optional[float], is_int: bool = False, use_latex: bool = True, float_precision=4
) -> str:
    if val_mean is None:
        return ""
    if is_int:
        mean_str = f"{int(round(val_mean))}"
        std_str = f"{int(round(val_std))}" if val_std is not None else "0"
    else:
        if float_precision == 0:
            mean_round = round(val_mean)
            std_round = round(val_std)
            mean_str = f"{mean_round}"
            std_str = f"{std_round}" if val_std is not None else "0"
        else:
            mean_round = round(val_mean, float_precision)
            std_round = round(val_std, float_precision)
            mean_str = f"{mean_round}".rstrip("0").rstrip(".")
            std_str = f"{std_round}".rstrip("0").rstrip(".") if val_std is not None else "0"

    if use_latex:
        return f"{mean_str} {{" + "\\small $\\pm$ " + f"{std_str}}}"

    if float(std_str) == 0:
        return mean_str

    return f"{mean_str} ± {std_str}"
