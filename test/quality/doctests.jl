# The docstring and manual doctests, as the Doctests job of `CI.yml` runs them.
#
# Documenter evaluates a page's `@meta` block in `Main`, and this file runs in a module of its own,
# so the package is imported into `Main` first.

using Documenter
using GMLDatasets

@eval Main import GMLDatasets

DocMeta.setdocmeta!(GMLDatasets, :DocTestSetup, :(using GMLDatasets); recursive = true)

doctest(GMLDatasets)
