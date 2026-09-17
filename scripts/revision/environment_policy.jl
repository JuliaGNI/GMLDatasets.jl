const MINIMUM_JULIA_VERSION = v"1.11.0"
const REQUIRED_GML_VERSION = v"0.7.0"
const REQUIRED_GO_VERSION = v"0.7.0"
const REQUIRED_NNP_VERSION = v"0.3.0"

"""Validate the reviewed dependency stack used by the revision experiments."""
function validate_environment_versions(
        julia_version::VersionNumber,
        gml_version::VersionNumber,
        go_version::VersionNumber,
        nnp_version::VersionNumber,
    )
    mismatches = String[]

    julia_version < MINIMUM_JULIA_VERSION && push!(
        mismatches,
        "Julia must be at least $MINIMUM_JULIA_VERSION (found $julia_version)",
    )
    gml_version == REQUIRED_GML_VERSION || push!(
        mismatches,
        "GeometricMachineLearning must be $REQUIRED_GML_VERSION (found $gml_version)",
    )
    go_version == REQUIRED_GO_VERSION || push!(
        mismatches,
        "GeometricOptimizers must be $REQUIRED_GO_VERSION (found $go_version)",
    )
    nnp_version == REQUIRED_NNP_VERSION || push!(
        mismatches,
        "NeuralNetworkParameters must be $REQUIRED_NNP_VERSION (found $nnp_version)",
    )

    isempty(mismatches) || error(
        "unsupported revision-experiment environment:\n  - " *
        join(mismatches, "\n  - ") *
        "\nInstantiate scripts/Manifest.toml from a clean checkout; do not run the " *
        "paper experiments with a substituted dependency stack.",
    )
    return nothing
end
