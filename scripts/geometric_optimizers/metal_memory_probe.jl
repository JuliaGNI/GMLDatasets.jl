# A bounded probe for the memory growth that `mnist_metal.jl` shows. It runs a handful of
# gradient steps at a *small* batch size and reports, per step, how much memory Metal has
# allocated and how large the Julia process has become. A watchdog task aborts the process
# before it can hurt the machine.
#
#   julia --project=scripts -e 'using Pkg; Pkg.add("Metal")'   # once, on the Mac; see mnist_metal.jl
#   julia --project=scripts scripts/geometric_optimizers/metal_memory_probe.jl [batch_size] [steps]

using LinearAlgebra: norm
using NNlib: batched_mul, batched_transpose, softmax, BatchedAdjOrTrans
using LinearAlgebra: Adjoint, Transpose
using Printf: @printf
import Metal, Random, Zygote

const batch_size = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 256
const steps = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 12
const gc_mode = length(ARGS) ≥ 3 ? Symbol(ARGS[3]) : :none    # :none, :incremental or :full
const rss_limit = 8 * 1024^3        # abort the process above this resident size

# ------------------------------------------------------------------------- watchdog ---

rss() = parse(Int, read(`ps -o rss= -p $(getpid())`, String) |> strip) * 1024

Threads.@spawn while true
    sleep(0.25)
    if rss() > rss_limit
        @error "watchdog: resident size exceeded the limit, aborting" rss = rss() / 1024^3
        flush(stderr)
        ccall(:_exit, Cvoid, (Cint,), 1)
    end
end

# --------------------------------------------------------------------------- device ---

const use_metal = Metal.functional()
const to_device = use_metal ? Metal.mtl : identity
use_metal || @warn "no Metal device — running on the host"

device_allocated() = use_metal ? Int(Metal.device().currentAllocatedSize) : 0
working_set() = use_metal ? Int(Metal.device().recommendedMaxWorkingSetSize) : 0

# ------------------------------------------------------- the same network as the script ---

const patch_length = 7
const n_heads = 7
const L = 16
const T = Float32
const dim = patch_length^2
const seq_length = (28 ÷ patch_length)^2
const n_classes = 10
const Dₕ = dim ÷ n_heads

function mat_tensor_mul(A::AbstractMatrix, x::AbstractArray{<:Number, 3})
    reshape(A * reshape(x, size(x, 1), :), size(A, 1), size(x, 2), size(x, 3))
end

_dense(Δ::AbstractArray) = Δ
_dense(Δ::BatchedAdjOrTrans) = permutedims(parent(Δ), (2, 1, 3))
_dense(Δ::Union{Adjoint, Transpose}) = permutedims(parent(Δ), (2, 1))

Zygote.@adjoint function mat_tensor_mul(A::AbstractMatrix, x::AbstractArray{<:Number, 3})
    function mat_tensor_mul_pullback(Δ)
        Δ₂ = reshape(_dense(Δ), size(A, 1), :)
        x₂ = reshape(x, size(x, 1), :)
        Δ₂ * transpose(x₂), reshape(transpose(A) * Δ₂, size(x))
    end
    mat_tensor_mul(A, x), mat_tensor_mul_pullback
end

function predict(ps::NamedTuple, input::AbstractArray{<:Number, 3})
    x = input
    for l in 1:L
        heads = ntuple(n_heads) do h
            i = (l - 1) * n_heads + h
            Q = mat_tensor_mul(transpose(ps.Q[i]), x)
            K = mat_tensor_mul(transpose(ps.K[i]), x)
            V = mat_tensor_mul(transpose(ps.V[i]), x)
            batched_mul(V, softmax(batched_mul(batched_transpose(Q), K) ./ sqrt(T(dim)); dims = 1))
        end
        y = reduce(vcat, heads)
        x = y + tanh.(mat_tensor_mul(ps.Wres[l], y) .+ ps.bres[l])
    end
    softmax(ps.Wclass * x[:, end, :]; dims = 1)
end

network_loss(ps, input, output) = norm(predict(ps, input) - output) / norm(output)

# ----------------------------------------------------------------------------- setup ---

rng = Random.Xoshiro(1234)
glorot(size...) = sqrt(T(24) / sum(size)) * (rand(rng, T, size...) .- T(0.5))

ps = (Q = [to_device(glorot(dim, Dₕ)) for _ in 1:(L * n_heads)],
    K = [to_device(glorot(dim, Dₕ)) for _ in 1:(L * n_heads)],
    V = [to_device(glorot(dim, Dₕ)) for _ in 1:(L * n_heads)],
    Wres = [to_device(glorot(dim, dim)) for _ in 1:L],
    bres = [to_device(zeros(T, dim)) for _ in 1:L],
    Wclass = to_device(glorot(n_classes, dim)))

input = to_device(rand(rng, T, dim, seq_length, batch_size))
output = to_device(rand(rng, T, n_classes, batch_size))

if use_metal
    println("device: ", Metal.device().name)
    @printf("recommended working set: %.1f GB\n", working_set() / 1024^3)
end
@printf("batch size %d, %d steps, watchdog at %.1f GB resident\n\n", batch_size, steps,
    rss_limit / 1024^3)
@printf("%5s  %12s  %12s  %10s  %10s\n", "step", "metal alloc", "process rss", "julia heap",
    "seconds")

function one_step(ps, input, output)
    ∂ps = Zygote.gradient(p -> network_loss(p, input, output), ps)[1]
    # force a synchronization, the way the real script does when it copies the gradient back
    Array(∂ps.Wclass)
    nothing
end

for step in 1:steps
    t = @elapsed begin
        if gc_mode === :pool || gc_mode === :poolgc
            Metal.@autoreleasepool one_step(ps, input, output)
        else
            one_step(ps, input, output)
        end
        gc_mode === :sync && Metal.device_synchronize()
        gc_mode === :incremental && GC.gc(false)
        gc_mode === :full && GC.gc(true)
        gc_mode === :poolgc && GC.gc(false)
    end
    @printf("%5d  %9.2f GB  %9.2f GB  %7.2f GB  %10.2f\n", step,
        device_allocated() / 1024^3, rss() / 1024^3,
        Base.gc_live_bytes() / 1024^3, t)
    flush(stdout)
end

println("\nafter an explicit GC:")
GC.gc(true)
use_metal && Metal.device_synchronize()
@printf("  metal alloc %.2f GB, process rss %.2f GB\n", device_allocated() / 1024^3,
    rss() / 1024^3)
