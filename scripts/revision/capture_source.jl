#!/usr/bin/env julia

# Capture one repository's exact state into a run bundle: its commit, whether the tree was dirty,
# and a reproducible patch that includes untracked files.
#
# There used to be two implementations of this — `capture_source` here and `capture_repository()` in
# `run_experiments.sh`, the same algorithm in two languages, each carrying the same comment about
# `git diff --no-index` exiting 1. Two copies of a provenance mechanism is the one kind of
# duplication that actively undermines the thing it exists to provide, and they were not even
# equivalent: only this one computes the SHA-256 and cross-checks the dirty flag against the
# captured patch. This CLI is what the runner calls for *both* repositories, so the bundle's
# `gmldatasets` capture now gets those two checks as well.
#
#   julia --project=scripts scripts/revision/capture_source.jl \
#       --repo /path/to/repo --prefix geometricoptimizers --out results/revision/<stamp>
#
# It writes `<out>/<prefix>.patch` and `<out>/<prefix>.status`, and prints four `key=value` lines — `sha`, `dirty`, `patch_file`
# and `patch_sha256` — which is a shell-readable form the runner reads with `read`, and which the
# retraction stage passes straight to the upstream benchmark's `--source-*` options.

include("arguments.jl")
include("records.jl")
using .RunRecords: capture_source

const USAGE = """usage: capture_source.jl --repo DIR --prefix NAME --out DIR

Write `<out>/<prefix>.patch` and `<out>/<prefix>.status`, and print the captured identity as
`key=value` lines:

  sha           the commit at HEAD
  dirty         whether the working tree had uncommitted or untracked changes
  patch_file    the patch's name, relative to --out
  patch_sha256  the patch's SHA-256, which is what a validator re-checks it against
"""

function main(args = ARGS)
    options = parse_arguments(args, (repo = "", prefix = "", out = ""); usage = USAGE)
    options === nothing && return 0

    isempty(options.repo) && throw(ArgumentError("--repo is required"))
    isempty(options.prefix) && throw(ArgumentError("--prefix is required"))
    isempty(options.out) && throw(ArgumentError("--out is required"))
    isdir(options.repo) || throw(ArgumentError("repository does not exist: $(options.repo)"))
    occursin(r"^[A-Za-z0-9._-]+$", options.prefix) ||
        throw(ArgumentError("--prefix may contain only letters, digits, dots, underscores and hyphens"))

    patch_file = "$(options.prefix).patch"
    source = capture_source(options.repo, joinpath(options.out, patch_file))
    # The porcelain listing is not part of the identity -- the patch is -- but it is what a human
    # reads first when a bundle says a tree was dirty.
    write(joinpath(options.out, "$(options.prefix).status"), source.status)
    println("sha=", source.sha)
    println("dirty=", source.dirty)
    println("patch_file=", patch_file)
    println("patch_sha256=", source.patch_sha256)
    0
end

try
    exit(main())
catch error
    println(stderr, "source capture failed: ", sprint(showerror, error))
    exit(1)
end
