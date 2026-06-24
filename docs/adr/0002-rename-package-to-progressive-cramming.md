# 2. Rename the package `compression_horizon` → `progressive_cramming`

Date: 2026-06-23

## Status
Accepted

## Context
The internal Python package is named `compression_horizon`. The public repository, the
paper ("Progressive Cramming"), and the GitHub Pages site are all named
`progressive_cramming`. The default migration path — copy the package as-is — would ship
a public repo whose `import compression_horizon` statements contradict the repo and paper
name in every README example, confusing readers and implying a different project.

## Decision
Rename the package to `progressive_cramming` during the copy: move
`src/compression_horizon/` → `src/progressive_cramming/`, rewrite all internal imports,
and update `pyproject.toml` (`name`, wheel package path, entry points) so the repo,
package, and README all agree.

## Consequences
- Clean public identity: repo == package == paper name; README examples are copy-pasteable.
- One-time mechanical churn: every `compression_horizon` import in the copied files must be
  rewritten; a stray reference will break `pip install -e .` (guarded by an acceptance
  criterion + grep).
- Hard to reverse once published: external users will `import progressive_cramming`, so the
  name is effectively permanent after release.
- The internal `compression_horizon` repo is untouched and keeps its name.
