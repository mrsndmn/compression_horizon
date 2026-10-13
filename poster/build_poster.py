#!/usr/bin/env python3
"""Render the poster offline, convert to FOGRA39 CMYK, and verify print requirements.

Dependencies: playwright (+ Chromium), pikepdf, pymupdf, Pillow, Ghostscript.
Run: python poster/build_poster.py [--ghostscript /path/to/gs]
"""
import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import pikepdf
import pymupdf as fitz
from PIL import ImageCms
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
WIDTH_MM, HEIGHT_MM = 594, 840  # The organizer explicitly requests 840, not ISO A1's 841.
ICC = ROOT / "profiles" / "FOGRA39L_coated.icc"


def render(path):
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(
            viewport={
                "width": round(WIDTH_MM / 25.4 * 96),
                "height": round(HEIGHT_MM / 25.4 * 96),
            }
        )
        page.emulate_media(media="print")
        # All fonts, formulas, logos, figures, and the QR are local.
        page.route("http://**/*", lambda route: route.abort())
        page.route("https://**/*", lambda route: route.abort())
        page.goto((ROOT / "poster.html").as_uri(), wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        page.wait_for_timeout(300)
        layout = page.evaluate(
            """() => {
          const bottom = e => e.getBoundingClientRect().bottom;
          const cols = [...document.querySelectorAll('[data-measure-role="column"]')];
          const ends = cols.map(c => bottom(c.querySelector('[data-measure-role="card"]:last-child')));
          const footer = document.querySelector('[data-measure-role="footer"]').getBoundingClientRect().top;
          const gaps = cols.flatMap(c => [...c.children].slice(1).map((e,i) => e.getBoundingClientRect().top - bottom(c.children[i])));
          const bounds = document.querySelector('.poster').getBoundingClientRect();
          const overflow = [...document.querySelectorAll('.card, .header, .footer, img, table, h1')].filter(e => {
            const r = e.getBoundingClientRect();
            return r.left < bounds.left - 1 || r.right > bounds.right + 1 || r.bottom > bounds.bottom + 1;
          }).map(e => e.className || e.tagName);
          return {spread_px: Math.max(...ends)-Math.min(...ends),
                  footer_gaps_px: ends.map(y => footer-y), card_gaps_px: gaps,
                  broken_images: [...document.images].filter(i => !i.complete || !i.naturalWidth).map(i => i.src),
                  overflow};
        }"""
        )
        assert layout["spread_px"] < 5, layout
        assert all(30 <= x <= 50 for x in layout["footer_gaps_px"]), layout
        assert all(12 <= x <= 50 for x in layout["card_gaps_px"]), layout
        assert not layout["broken_images"] and not layout["overflow"], layout
        page.pdf(path=str(path), prefer_css_page_size=True, print_background=True)
        browser.close()
        return layout


def convert(gs, rgb, converted, final):
    subprocess.run(
        [
            gs,
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.6",
            "-r299",
            "-sColorConversionStrategy=CMYK",
            "-sProcessColorModel=DeviceCMYK",
            f"-sOutputICCProfile={ICC}",
            "-dOverrideICC",
            "-dRenderIntent=1",
            "-dDownsampleColorImages=true",
            "-dColorImageResolution=299",
            "-dColorImageDownsampleThreshold=1.0",
            "-dColorImageDownsampleType=/Bicubic",
            "-dDownsampleGrayImages=true",
            "-dGrayImageResolution=299",
            "-dGrayImageDownsampleThreshold=1.0",
            "-dDownsampleMonoImages=true",
            "-dMonoImageResolution=299",
            "-dMonoImageDownsampleThreshold=1.0",
            "-dAutoFilterColorImages=false",
            "-dColorImageFilter=/FlateEncode",
            "-dAutoFilterGrayImages=false",
            "-dGrayImageFilter=/FlateEncode",
            "-dEmbedAllFonts=true",
            f"-sOutputFile={converted}",
            str(rgb),
        ],
        check=True,
    )
    with pikepdf.open(converted) as pdf:
        assert len(pdf.pages) == 1, "Poster must be exactly one page"
        # Chromium quantizes CSS mm to print pixels. Normalize subpixel rounding.
        pdf.pages[0].MediaBox = [0, 0, WIDTH_MM / 25.4 * 72, HEIGHT_MM / 25.4 * 72]
        pdf.pages[0].CropBox = pdf.pages[0].MediaBox
        profile = pdf.make_stream(ICC.read_bytes())
        profile.N = 4
        profile.Alternate = pikepdf.Name("/DeviceCMYK")
        intent = pikepdf.Dictionary(
            Type=pikepdf.Name("/OutputIntent"),
            S=pikepdf.Name("/GTS_PDFX"),
            OutputConditionIdentifier="FOGRA39",
            OutputCondition="Coated FOGRA39 (ISO 12647-2:2004)",
            RegistryName="http://www.color.org",
            Info="Coated FOGRA39 (ISO 12647-2:2004)",
            DestOutputProfile=profile,
        )
        pdf.Root.OutputIntents = [pdf.make_indirect(intent)]
        pdf.docinfo["/Title"] = "Progressive Cramming — portrait print poster"
        pdf.save(final)


def verify(final):
    with pikepdf.open(final) as pdf:
        assert len(pdf.pages) == 1
        intent = pdf.Root.OutputIntents[0]
        assert intent.OutputConditionIdentifier == "FOGRA39"
        assert intent.DestOutputProfile.N == 4
        assert intent.DestOutputProfile.read_bytes() == ICC.read_bytes()
        for obj in pdf.objects:
            if isinstance(obj, pikepdf.Stream) and obj.get("/Subtype") == pikepdf.Name("/Image"):
                if not obj.get("/ImageMask", False):
                    assert obj.get("/ColorSpace") in (pikepdf.Name("/DeviceCMYK"), pikepdf.Name("/DeviceGray")), obj
    doc = fitz.open(final)
    page = doc[0]
    for actual, mm in zip([page.rect.width, page.rect.height], [WIDTH_MM, HEIGHT_MM]):
        assert math.isclose(actual * 25.4 / 72, mm, abs_tol=0.01)
    images = []
    for im in page.get_images(full=True):
        xref, _, width, height = im[:4]
        for rect in page.get_image_rects(xref):
            dpi = [width * 72 / rect.width, height * 72 / rect.height]
            assert all(150 <= d <= 300 for d in dpi), (xref, dpi)
            images.append({"xref": xref, "dpi": [round(d, 2) for d in dpi]})
    assert images, "No raster images found"
    text = page.get_text()
    assert "Dmitrii Tarasov" in text and "Progressive Cramming" in text
    assert "$" not in text and "TODO" not in text
    # Preview is rendered from the delivered CMYK PDF, not from the RGB HTML.
    pix = page.get_pixmap(matrix=fitz.Matrix(1600 / page.rect.width, 1600 / page.rect.width), colorspace=fitz.csRGB)
    pix.save(ROOT / "poster_preview.png")
    report = {
        "file": final.name,
        "pages": len(doc),
        "width_mm": WIDTH_MM,
        "height_mm": HEIGHT_MM,
        "orientation": "portrait",
        "color_model": "CMYK",
        "icc_description": ImageCms.getProfileDescription(ImageCms.getOpenProfile(str(ICC))).strip(),
        "output_condition": "FOGRA39",
        "images": images,
    }
    doc.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ghostscript", default="gs")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="poster-print-") as tmp:
        rgb, converted = Path(tmp) / "rgb.pdf", Path(tmp) / "cmyk.pdf"
        layout = render(rgb)
        convert(args.ghostscript, rgb, converted, ROOT / "poster.pdf")
    report = verify(ROOT / "poster.pdf")
    report["layout"] = layout
    (ROOT / "print_report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Keep the historical preview filename consistent with the final artifact.
    (ROOT / "poster_preview.pdf").write_bytes((ROOT / "poster.pdf").read_bytes())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
