using Test

include("run_artifact_schema.jl")
using .RunArtifactSchema: CONFIGURATION_NAMES, IMAGE_LOSS_HEADER, IMAGE_RECORD_HEADER,
                          PENDULUM_RECORD_HEADER, STAGE_HEADER, validate_image_artifacts,
                          validate_pendulum_artifacts, validate_stage_table

csv_field(value) = begin
    text = string(value)
    occursin(r"[\",\r\n]", text) ? "\"$(replace(text, '\"' => "\"\""))\"" : text
end

function write_table(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            println(io, join((csv_field(row[field]) for field in header), ','))
        end
    end
end

function image_record(configuration_key, repetition, seed; status="ok")
    Dict(
        "schema_version" => 4,
        "dataset" => "mnist",
        "configuration_key" => configuration_key,
        "configuration" => CONFIGURATION_NAMES[configuration_key],
        "optimizer_role" => configuration_key == "geometric-adam-cayley" ? "proposed" :
                            "riemannian-adam-baseline",
        "learning_rate" => 0.001,
        "retraction" => "cayley",
        "second_moment" => configuration_key == "geometric-adam-cayley" ?
                           "coordinate-wise" : "scalar (quotient norm)",
        "transport" => "global-section",
        "repetition" => repetition,
        "seed" => seed,
        "status" => status,
        "epochs_completed" => 2,
        "final_loss" => 0.8,
        "best_loss" => 0.8,
        "test_accuracy" => 0.25,
        "total_seconds" => 1.0,
        "seconds_per_epoch" => 0.5,
        "timed_steps" => 2,
        "gradient_ad_seconds_total" => 0.4,
        "gradient_ad_seconds_per_step" => 0.2,
        "optimizer_state_direction_seconds_total" => 0.2,
        "optimizer_state_direction_seconds_per_step" => 0.1,
        "retraction_application_seconds_total" => 0.1,
        "retraction_application_seconds_per_step" => 0.05,
        "peak_device_bytes" => 0,
        "backend" => "cpu",
        "message" => status == "ok" ? "ok" : "FAILED: short smoke",
    )
end

@testset "revision run artifact schemas" begin
    mktempdir() do directory
        configurations = ["geometric-adam-cayley", "scalar-moment-adam"]
        seeds = [1234, 1235]
        records = [image_record(configuration, repetition, seed)
                   for configuration in configurations
                   for (repetition, seed) in enumerate(seeds)]
        records_path = joinpath(directory, "mnist-runs.csv")
        write_table(records_path, IMAGE_RECORD_HEADER, records)

        losses = Dict{String,Any}[]
        run = 0
        for configuration in configurations
            for repetition in eachindex(seeds)
                run += 1
                for step in 1:2
                    push!(losses, Dict(
                        "run" => run,
                        "configuration" => CONFIGURATION_NAMES[configuration],
                        "repetition" => repetition,
                        "epoch" => step,
                        "batch" => 1,
                        "step" => step,
                        "loss" => 1 - step / 10,
                    ))
                end
            end
        end
        losses_path = joinpath(directory, "mnist-losses.csv")
        write_table(losses_path, IMAGE_LOSS_HEADER, losses)

        summary = validate_image_artifacts(records_path, losses_path; dataset="mnist", seeds,
            configurations, expected_epochs=2, expected_backend="cpu")
        @test summary.records == 4
        @test summary.losses == 8
        @test summary.statuses == Dict("ok" => 4)

        losses[2]["step"] = 3
        write_table(losses_path, IMAGE_LOSS_HEADER, losses)
        @test_throws ArgumentError validate_image_artifacts(records_path, losses_path;
            dataset="mnist", seeds, configurations, expected_epochs=2, expected_backend="cpu")
        losses[2]["step"] = 2
        write_table(losses_path, IMAGE_LOSS_HEADER, losses)

        records[1]["status"] = "failed_validation"
        write_table(records_path, IMAGE_RECORD_HEADER, records)
        @test_throws ArgumentError validate_image_artifacts(records_path, losses_path;
            dataset="mnist", seeds, configurations, expected_epochs=2, expected_backend="cpu")
        @test validate_image_artifacts(records_path, losses_path; dataset="mnist", seeds,
            configurations, expected_epochs=2, expected_backend="cpu",
            allow_validation_failures=true).statuses["failed_validation"] == 1

        checkpoints = String[]
        pendulum_records = Dict{String,Any}[]
        for (repetition, seed) in enumerate(seeds)
            checkpoint = joinpath(directory, "pendulum-seed-$seed.h5")
            write(checkpoint, "fixture")
            push!(checkpoints, checkpoint)
            push!(pendulum_records, Dict(
                "schema_version" => 1,
                "dataset" => "pendulum",
                "configuration" => "geometric-adam",
                "repetition" => repetition,
                "seed" => seed,
                "status" => "ok",
                "epochs_completed" => 2,
                "final_loss" => 0.4,
                "best_loss" => 0.3,
                "total_seconds" => 1.2,
                "seconds_per_epoch" => 0.6,
                "host_allocated_bytes" => 1024,
                "gc_seconds" => 0.1,
                "backend" => "CPU",
                "checkpoint" => checkpoint,
            ))
        end
        pendulum_path = joinpath(directory, "pendulum-runs.csv")
        write_table(pendulum_path, PENDULUM_RECORD_HEADER, pendulum_records)
        @test validate_pendulum_artifacts(pendulum_path, directory; seeds,
            expected_epochs=2, expected_backend="cpu").records == 2

        write_table(pendulum_path, PENDULUM_RECORD_HEADER, pendulum_records[1:1])
        @test validate_pendulum_artifacts(pendulum_path, directory; seeds,
            expected_epochs=2, expected_backend="cpu", allow_partial=true).records == 1
        @test_throws ArgumentError validate_pendulum_artifacts(pendulum_path, directory; seeds,
            expected_epochs=2, expected_backend="cpu")

        stages_path = joinpath(directory, "stages.csv")
        stage_rows = [
            Dict("stage" => "mnist", "status" => "failed:1",
                "started_utc" => "2026-08-31T00:00:00Z",
                "finished_utc" => "2026-08-31T00:00:01Z", "command" => "first"),
            Dict("stage" => "mnist", "status" => "ok",
                "started_utc" => "2026-08-31T00:00:02Z",
                "finished_utc" => "2026-08-31T00:00:03Z", "command" => "resume, second"),
        ]
        write_table(stages_path, STAGE_HEADER, stage_rows)
        @test validate_stage_table(stages_path, ["mnist"]).passed == 1
        @test_throws ArgumentError validate_stage_table(stages_path, ["fashion-mnist"])
    end
end
