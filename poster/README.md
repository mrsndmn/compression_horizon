The print poster is `poster.pdf`: one portrait page, **594 × 840 mm**, CMYK with
an embedded **FOGRA39L Coated** output profile. The organizer explicitly specifies
840 mm height (standard ISO A1 is 841 mm). `poster_preview.pdf` is an identical
copy; `poster_preview.png` is rendered from the final CMYK PDF for screen review.
The compact header places institution logos beside the venue badge and a 70 mm
QR beside the title. Main copy is about 25 pt; captions and tables are 21 pt. The build enforces minimum
24 pt body copy and 20 pt captions/tables. Five figures use whitespace-trimmed
versions (`images/*_trimmed.png`); the unmodified source images are retained.
Redundant projection/scoring diagrams and repeated conclusion copy are omitted.
Text and SVG logos remain vector; raster elements are between 150 and 300 DPI
(with subpixel rounding at the upper bound).

Edit `poster.html`, then rebuild from any working directory:

```sh
python /path/to/compression_horizon/poster/build_poster.py
# If Ghostscript is outside PATH:
python poster/build_poster.py --ghostscript /path/to/gs
```

Dependencies: `playwright` with its Chromium browser, `pikepdf`, `pymupdf`,
`Pillow`, and Ghostscript. The build needs no network access. Fonts, logos,
figures, QR code, and ICC profile are committed locally. The builder validates
column alignment, footer/card gaps, broken images, page overflow, page size,
page count, the embedded profile, image color spaces, effective raster DPI,
and minimum physical font sizes.
It writes the measured results to `print_report.json`.

Layout adapted with the `paper-to-poster` skill's portrait two-column workflow,
retaining the existing AXXX palette and research figures. Its core `preflight`,
`measure`, and `verify-final` checks pass. The title intentionally aligns to the
left body edge beside the enlarged QR; `polish --strict --title-offset-max 0.07`
allows that layout (the default 0.06 assumes a centered title). The optional
style/provenance gates are not used for this imported poster.

The full/progressive comparison is aligned with
`paper/tables/full_vs_progressive.tex`; family capacity values come from
`paper/tables/progressive_modifications.tex` (means over 50 PG19 samples).
Downstream results come from `paper/tables/semantic_evaluation.tex`; the MMLU
claim is described in `paper/example_paper.tex` and
`paper/tables/manual/rebuttle_mmlu.tex`.

Bundled Inter fonts were retrieved from Google's Inter font distribution and
are licensed under the SIL Open Font License (`fonts/LICENSE.txt`). The
FOGRA39 profile is from CTAN's `support/colorprofiles/FOGRA39L_coated.icc`:
https://ctan.math.washington.edu/tex-archive/support/colorprofiles/
See `profiles/LICENSE.txt` for its distribution terms.
