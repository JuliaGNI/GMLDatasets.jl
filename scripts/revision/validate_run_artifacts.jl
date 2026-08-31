#!/usr/bin/env julia

include("run_artifact_schema.jl")
using .RunArtifactSchema: normalize_configurations, split_list, validate_image_artifacts,
                          validate_pendulum_artifacts, validate_run_artifacts

function usage(io::IO=stdout)
    println(io, """usage: validate_run_artifacts.jl --run-dir DIR [options]

Validate one complete runner directory:
  --mode smoke|full
  --stages LIST
  --seeds LIST
  --configurations LIST
  --image-epochs N
  --pendulum-epochs N
  --backend cpu|cuda
  --retraction-repo DIR
  --allow-validation-failures

Focused validation used by restart logic:
  --image DATASET [--artifact-prefix PREFIX]
  --pendulum [--allow-partial]
""")
end

function parse_integer_argument(option, value; minimum=0)
    parsed = tryparse(Int, value)
    parsed === nothing && throw(ArgumentError("$option requires an integer, got $value"))
    parsed >= minimum || throw(ArgumentError("$option must be at least $minimum, got $parsed"))
    parsed
end

function main(args=ARGS)
    run_dir = ""
    mode = "smoke"
    stages_value = "mnist,fashion-mnist,pendulum,retraction"
    seeds_value = "1234"
    configurations_value = "all"
    image_epochs = 2
    pendulum_epochs = 2
    backend = "cpu"
    retraction_repo = ""
    image = ""
    artifact_prefix = ""
    pendulum = false
    allow_partial = false
    allow_validation_failures = false

    index = 1
    while index <= length(args)
        argument = args[index]
        if argument in ("-h", "--help")
            usage()
            return 0
        elseif argument in ("--run-dir", "--mode", "--stages", "--seeds", "--configurations",
                "--image-epochs", "--pendulum-epochs", "--backend", "--retraction-repo",
                "--image", "--artifact-prefix")
            index == length(args) && throw(ArgumentError("missing value for $argument"))
            value = args[index + 1]
            index += 2
            if argument == "--run-dir"
                run_dir = value
            elseif argument == "--mode"
                mode = value
            elseif argument == "--stages"
                stages_value = value
            elseif argument == "--seeds"
                seeds_value = value
            elseif argument == "--configurations"
                configurations_value = value
            elseif argument == "--image-epochs"
                image_epochs = parse_integer_argument(argument, value; minimum=1)
            elseif argument == "--pendulum-epochs"
                pendulum_epochs = parse_integer_argument(argument, value; minimum=1)
            elseif argument == "--backend"
                backend = lowercase(value)
            elseif argument == "--retraction-repo"
                retraction_repo = value
            elseif argument == "--artifact-prefix"
                artifact_prefix = value
            else
                image = lowercase(value)
            end
        elseif argument == "--pendulum"
            pendulum = true
            index += 1
        elseif argument == "--allow-partial"
            allow_partial = true
            index += 1
        elseif argument == "--allow-validation-failures"
            allow_validation_failures = true
            index += 1
        else
            throw(ArgumentError("unknown argument: $argument"))
        end
    end

    isempty(run_dir) && throw(ArgumentError("--run-dir is required"))
    run_dir = abspath(run_dir)
    isdir(run_dir) || throw(ArgumentError("run directory does not exist: $run_dir"))
    seeds = [parse_integer_argument("--seeds", value) for value in split_list(seeds_value)]
    configurations = normalize_configurations(configurations_value)
    backend in ("cpu", "cuda") || throw(ArgumentError("--backend must be cpu or cuda"))
    !isempty(image) && pendulum && throw(ArgumentError("select at most one focused validation"))

    if !isempty(image)
        isempty(artifact_prefix) && (artifact_prefix = image)
        summary = validate_image_artifacts(
            joinpath(run_dir, "$artifact_prefix-runs.csv"),
            joinpath(run_dir, "$artifact_prefix-losses.csv");
            dataset=image, seeds, configurations, expected_epochs=image_epochs,
            expected_backend=backend, allow_validation_failures)
        println("validated $image artifacts: $(summary.records) records, $(summary.losses) loss rows")
    elseif pendulum
        summary = validate_pendulum_artifacts(joinpath(run_dir, "pendulum-runs.csv"), run_dir;
            seeds, expected_epochs=pendulum_epochs, expected_backend=backend, allow_partial)
        println("validated pendulum artifacts: $(summary.records) records")
    else
        stages = lowercase.(split_list(stages_value))
        if "retraction" in stages
            isempty(retraction_repo) && throw(ArgumentError(
                "--retraction-repo is required when the retraction stage is selected"))
            isdir(retraction_repo) || throw(ArgumentError(
                "retraction repository does not exist: $retraction_repo"))
        end
        summaries = validate_run_artifacts(run_dir; mode=lowercase(mode), stages, seeds,
            configurations, expected_image_epochs=image_epochs,
            expected_pendulum_epochs=pendulum_epochs, expected_backend=backend,
            retraction_repo, allow_validation_failures)
        println("validated run artifacts: ", join(summaries, "; "))
    end
    0
end

try
    exit(main())
catch exception
    println(stderr, "run artifact validation failed: ", sprint(showerror, exception))
    exit(1)
end
