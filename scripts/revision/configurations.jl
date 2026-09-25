# The optimizer configurations of the comparison, by stable key.
#
# One table, because three copies of it used to exist — in the image trainer, in the pendulum
# trainer and in the validator that checks the other two agree with it. The display name is what
# every record and every report prints, so a second copy is a silently divergent results table.
# The validators still compare a record's `configuration` against `CONFIGURATION_NAMES`; with one
# table that check no longer guards against drift between scripts, but it still rejects a
# hand-edited or concatenated CSV.
#
# What is *not* here is the optimizer object itself: the image trainer and the pendulum trainer
# build different ones from the same key, and both take their rates and their element type from
# their own environment.
#
# This file holds definitions only; `records.jl` is the module that includes it.

const CONFIGURATION_ORDER = [
    "geometric-adam-cayley",
    "scalar-moment-adam",
    "standard-adam",
    "gradient",
    "momentum"
]

# Unlike the transformer, an SAE cannot have an unconstrained Adam row without ceasing to be
# symplectic: its PSD layers must keep their Stiefel weights. The pendulum comparison is therefore
# these four intrinsic configurations.
const PENDULUM_CONFIGURATION_ORDER = [
    "geometric-adam-cayley",
    "scalar-moment-adam",
    "gradient",
    "momentum"
]

const CONFIGURATION_ALIASES = Dict(
    "adam-stiefel" => "geometric-adam-cayley",
    "adam-regular" => "standard-adam"
)

"""
The metadata every run record quotes for a configuration: the display name, the role the
comparison gives it, and the three method properties that distinguish the rows.

`second_moment` is the default. The image trainer overrides it for `scalar-moment-adam` when
`MNIST_SCALAR_MOMENT_ADAM_AMBIENT_NORM` selects the ambient norm of `li2020efficient`
Algorithm 2 over the `GeometricOptimizers` quotient-space norm.
"""
const CONFIGURATIONS = Dict(
    "geometric-adam-cayley" => (name = "Geometric Adam (Stiefel, Cayley retraction)",
        role = "proposed", retraction = "cayley", second_moment = "coordinate-wise",
        transport = "global-section"),
    "scalar-moment-adam" => (name = "Scalar Moment Adam (Stiefel, Cayley retraction)",
        role = "riemannian-adam-baseline", retraction = "cayley",
        second_moment = "scalar (quotient norm)", transport = "global-section"),
    "standard-adam" => (name = "Standard Adam (unconstrained)",
        role = "non-geometric-ablation", retraction = "none",
        second_moment = "coordinate-wise", transport = "none"),
    "gradient" => (name = "Riemannian gradient (Stiefel, Cayley retraction)",
        role = "diagnostic", retraction = "cayley", second_moment = "none",
        transport = "none"),
    "momentum" => (name = "Riemannian momentum (Stiefel, Cayley retraction)",
        role = "diagnostic", retraction = "cayley", second_moment = "none",
        transport = "global-section")
)

const CONFIGURATION_NAMES = Dict(key => entry.name for (key, entry) in CONFIGURATIONS)

function split_list(value::AbstractString)
    filter(!isempty, strip.(split(value, ','; keepempty = false)))
end

"""
    normalize_configurations(value; order = CONFIGURATION_ORDER, what = "configuration")

Resolve a comma-separated selection — keys, the two legacy aliases, or `all` — into a subset of
`order`, in the order of `order`. `order` is `PENDULUM_CONFIGURATION_ORDER` for the pendulum
stage, which is what rejects `standard-adam` there.
"""
function normalize_configurations(value::AbstractString; order = CONFIGURATION_ORDER,
        what::AbstractString = "configuration")
    requested = [get(CONFIGURATION_ALIASES, lowercase(entry), lowercase(entry))
                 for entry in split_list(value)]
    "all" in requested && return copy(order)
    isempty(requested) && throw(ArgumentError("$what list is empty"))
    unknown = filter(key -> key ∉ order, requested)
    isempty(unknown) || throw(ArgumentError(
        "unknown $what(s): $(join(unknown, ", ")); choose $(join(order, ", "))"))
    filter(key -> key in requested, order)
end

function normalize_pendulum_configurations(value::AbstractString)
    normalize_configurations(
        value; order = PENDULUM_CONFIGURATION_ORDER, what = "pendulum configuration")
end
