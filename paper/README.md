# Paper

## Layout

```
paper/
├── example_paper.tex            # main manuscript
├── rebuttle.tex                 # rebuttal, \include'd from the main file
├── example_paper.bib            # bibliography
├── figures/                     # all .pdf figures and .tikz sources
├── styles/                      # ICML class + auxiliary .sty / .bst files
├── check_unused_attachments.py  # CI gate: fails on unreferenced attachments
├── .latexmkrc                   # sets TEXINPUTS for figures//, styles// and out_dir=build
└── build/                       # latexmk output (PDF + aux files, git-ignored)
```

## Build

From the repository root:

```
make paper
```

The target runs `check_unused_attachments.py` first, then invokes `latexmk -pdf`.
The compiled PDF is written to `paper/build/example_paper.pdf`; all aux files
stay inside `paper/build/`.

To clean every build artifact:

```
make paper-clean
```

## Adding a figure or style file

1. Drop the file into `paper/figures/` or `paper/styles/`.
2. Reference it from a `.tex` source — `\includegraphics{<basename>}`,
   `\input{<basename.tikz>}`, or `\usepackage{<basename>}`. No subdir prefix
   is needed because `paper/.latexmkrc` extends `TEXINPUTS` to cover both
   subdirectories.
3. Run `make paper`. The unused-attachments check will fail if you forget to
   reference the file (or, after removing a reference, forget to delete the
   file).
