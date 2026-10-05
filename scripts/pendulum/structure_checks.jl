# structure_checks.jl — the embedding checks of `branch_report.jl`, as functions
#
# Included by `branch_report.jl`, which prints them per level, and by `train_sae.jl`, which runs them
# on every checkpoint so that a run can keep the weights that pass them. Why these three checks and
# not an accuracy number is explained in the header of `branch_report.jl`.
#
# Nothing here loads a network: every function takes the encoder as an argument.

using Printf

const pendulum_length = 1.0

lift(θ, pθ) = Float32[pendulum_length * sin(θ), pendulum_length * cos(θ),
    cos(θ) * pθ / pendulum_length, -sin(θ) * pθ / pendulum_length]
embed(enc, θ, p) = (Z = [enc(lift(a, b)) for (a, b) in zip(θ, p)];
    hcat([z[1] for z in Z], [z[2] for z in Z]))

# Level sets analytically, not integrated: the encoder is a pointwise map, so the latent curve is
# exactly the image of the level set, with no integrator error and no wrap-around artefact. The
# librating parametrisation sin(φ/2) = k sin s, φ = θ - π, k² = (1+H)/2 removes the sqrt singularity
# at the turning points.
librating(H, n) = (s = 2π .* range(0, 1; length = n + 1)[1:(end - 1)]; k = sqrt((1 + H) / 2);
    (π .+ 2 .* asin.(clamp.(k .* sin.(s), -1, 1)), 2 .* k .* cos.(s)))
rotating(H, n; sgn = -1) = (s = 2π .* range(0, 1; length = n + 1)[1:(end - 1)];
    (collect(s), sgn .* sqrt.(2 .* (H .- cos.(s)))))
"One branch of the separatrix: s ∈ [π/2,3π/2] gives ℓ₋, s ∈ [3π/2,5π/2] gives ℓ₊."
function sepbranch(lo, hi, n)
    k = sqrt((1 + (1 - 1e-12)) / 2)
    s = range(lo, hi; length = n)
    (π .+ 2 .* asin.(clamp.(k .* sin.(s), -1, 1)), 2 .* k .* cos.(s))
end

nxt(M, i) = mod1(i + 1, size(M, 1))
signed_area(M) = sum(M[i, 1] * M[nxt(M, i), 2] - M[nxt(M, i), 1] * M[i, 2] for i in 1:size(M, 1)) / 2
"Segment crossings of a closed polyline with itself. Zero iff the curve is simple."
function self_crossings(M)
    n = size(M, 1)
    c = 0
    for i in 1:(n - 1), j in (i + 2):n
        (i == 1 && j == n) && continue
        ax, ay = M[i, 1], M[i, 2]
        bx, by = M[i + 1, 1], M[i + 1, 2]
        cx, cy = M[j, 1], M[j, 2]
        dx, dy = M[nxt(M, j), 1], M[nxt(M, j), 2]
        d1 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        d2 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
        d3 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
        d4 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
        ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0)) && (c += 1)
    end
    c
end
function winding(M, x, y)
    t = 0.0
    for i in 1:size(M, 1)
        a = atan(M[i, 2] - y, M[i, 1] - x)
        b = atan(M[nxt(M, i), 2] - y, M[nxt(M, i), 1] - x)
        d = b - a
        t += d > π ? d - 2π : d < -π ? d + 2π : d
    end
    round(Int, t / 2π)
end

const HLIB = collect(range(-0.4, 0.9; length = 11))
const HROT = [1.001, 1.05, 1.14, 1.27, 1.40, 1.50, 1.60, 1.75, 1.90, 2.00, 2.50, 4.00, 8.00, 16.91]

"""
Walk a family in order and check each level's image. `frac` is the fraction of this level's image
inside the previous one: for an embedding every vertex of one curve lies on a single side of the
other, so it must come out 0% or 100%. Anything between is a crossing, and two disjoint orbits
with intersecting images is non-injectivity directly.

The orientation a family is traversed with is a convention -- the librating loops come out negative
here and the rotating ones positive -- so what an embedding forces is not a positive area but a
CONSTANT sign along the family. The first level's sign is the reference.

Returns one row per level and, for each check, the first level that fails it (`nothing` if none).
"""
function family_checks(enc, Hs, curve)
    rows = []
    prev = nothing
    ref = nothing
    firstbad = Dict{String, Union{Nothing, Float64}}(
        "sign" => nothing, "nested" => nothing, "simple" => nothing)
    for H in Hs
        θ, p = curve(H)
        M = embed(enc, θ, p)
        A = signed_area(M)
        sc = self_crossings(M)
        frac = prev === nothing ? NaN :
               count(winding(prev, M[i, 1], M[i, 2]) != 0 for i in 1:size(M, 1)) / size(M, 1)
        ref === nothing && (ref = sign(A))
        sign(A) != ref && firstbad["sign"] === nothing && (firstbad["sign"] = H)
        sc > 0 && firstbad["simple"] === nothing && (firstbad["simple"] = H)
        !isnan(frac) && !(frac in (0.0, 1.0)) && firstbad["nested"] === nothing &&
            (firstbad["nested"] = H)
        push!(rows, (; H, θ, p, A, sc, frac))
        prev = M
    end
    rows, firstbad
end

"""
    embedding_checks(enc; nsamp = 1600)

The structural verdict on the half of the cylinder the training grid covers: γ± with signed areas
of opposite sign, and the librating and `p_θ < 0` rotating families sign-constant, nested and simple
at every level. `ok` is true iff all of these hold. The `p_θ > 0` family is not checked: it is
outside the data, and `branch_report.jl` reports it separately.
"""
function embedding_checks(enc; nsamp = 1600)
    Am = signed_area(embed(enc, sepbranch(π / 2, 3π / 2, nsamp)...))
    Ap = signed_area(embed(enc, sepbranch(3π / 2, 5π / 2, nsamp)...))
    _, lib = family_checks(enc, HLIB, H -> librating(H, nsamp))
    _, low = family_checks(enc, HROT, H -> rotating(H, nsamp; sgn = -1))
    clean(d) = all(v === nothing for v in values(d))
    (; ok = sign(Am) != sign(Ap) && clean(lib) && clean(low), Am, Ap, lib, low)
end

failures(d) = isempty([v for v in values(d) if v !== nothing]) ? "ok" :
              join([@sprintf("%s from H = %.3f", k, v)
                    for (k, v) in sort([(k, v) for (k, v) in d if v !== nothing]; by = x -> x[2])], ", ")
"One line per checkpoint, for a training log."
checks_line(r) = @sprintf("%s  gamma- %+8.4f  gamma+ %+8.4f  librating limit %7.4f | librating: %s | p<0: %s",
    r.ok ? "PASS" : "fail", r.Am, r.Ap, abs(r.Am) - abs(r.Ap), failures(r.lib), failures(r.low))

"The angular action ∮ p_θ dθ, analytic, by the level set's own parametrisation."
function angular(H, n = 16_000)
    s = 2π .* range(0, 1; length = n + 1)[1:(end - 1)]
    if H < 1
        k = sqrt((1 + H) / 2)
        p = 2 .* k .* cos.(s)
        dθ = 2 .* k .* cos.(s) ./ sqrt.(max.(1 .- (k .* sin.(s)) .^ 2, eps()))
        abs(sum(p .* dθ) * (2π / n))
    else
        abs(sum(-sqrt.(2 .* (H .- cos.(s)))) * (2π / n))
    end
end

"""
    threshold_errors(enc, H_L, H_R; nsamp = 1600)

Relative action errors (|J_latent| - J_angular)/J_angular on the complete librating orbit at H_L and
the complete p_θ < 0 rotating orbit at H_R, and the threshold ε* = (J_L - J_R)/(J_L + J_R). An
encoder that embeds ℳ⁻ has at least one of the two errors of modulus ε* or more, whatever its weights.
"""
function threshold_errors(enc, H_L, H_R; nsamp = 1600)
    J_L, J_R = angular(H_L), angular(H_R)
    A_L = abs(signed_area(embed(enc, librating(H_L, nsamp)...)))
    A_R = abs(signed_area(embed(enc, rotating(H_R, nsamp; sgn = -1)...)))
    (; eL = (A_L - J_L) / J_L, eR = (A_R - J_R) / J_R, epsilon = (J_L - J_R) / (J_L + J_R))
end
threshold_line(t) = @sprintf("action error at H_L %+6.1f%%, at H_R %+6.1f%% (threshold %.1f%%)",
    100t.eL, 100t.eR, 100t.epsilon)
