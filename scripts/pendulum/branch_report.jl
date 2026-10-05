# branch_report.jl — the evaluation table every pendulum SAE run should be read through
#
# Takes any weights file and reports, per regime AND per branch of the cylinder:
#
#   J_angular   the true action, analytic
#   J_latent    the latent loop's enclosed area, which is exactly ∮ p·dq of the decoded loop
#               because the decoder is symplectic and globally defined on the plane
#   gap         |J_latent - J_angular| / J_angular
#   recon       relative L2 reconstruction error on that level set
#   nested      whether this level's image lies wholly on one side of the level before it
#   simple      whether the image is a Jordan curve
#
# Why per branch. Above the separatrix each energy carries two orbits, p_θ > 0 and p_θ < 0, and they
# are separate loops around the cylinder. The paper's training grid contains only the p_θ < 0 one
# (its momenta are the fractions {0, ±2/5, ±3/4, −1, −2, −3} of the separatrix momentum, so every
# |f| > 1 is negative). A number aggregated over both branches is meaningless: on a two-directional
# grid one of them is reproduced to a few percent and the other is ~93% off, and that is not a
# training deficiency. See notes/separatrix_action.tex, "The branch the data never visits".
#
# Why the structural columns. If the encoder embeds the cylinder then its image is an open annulus,
# its complement has one bounded hole K, and every rotating image encloses K. Then:
#
#   * each image is a Jordan curve                            (injective continuous image of a circle)
#   * consecutive images are nested                           (disjoint orbits have disjoint images)
#   * enclosed areas keep one sign along the family, with magnitude bounded below by area(K)
#
# Exactly one of the two rotating families nests outward and grows; the other nests inward towards K
# and SHRINKS. Shrinking is therefore not a failure -- it is required of whichever family the chart
# puts on the inside. Only the three checks above are violations, and each is a certificate, by
# contraposition, that the map is not an embedding there. They need no ground-truth action and cost
# one forward pass per level set, which is what makes them worth running on every evaluation: a
# reconstruction error tells you the fit is poor, these tell you the chart has stopped being a chart.
#
# Run from the repository root:
#
#   julia --project=scripts scripts/pendulum/branch_report.jl
#
#   SAE_WEIGHTS=/path/to/weights.h5 julia --project=scripts scripts/pendulum/branch_report.jl
#
# Moved here from the symplectic-autoencoder talk's working directory
# (SciCade26/simulation_results_for_talk/branch_report.jl).
#
# Environment:
#   SAE_WEIGHTS   <GML_OUTDIR>/pendulum_sae.h5   the file to evaluate
#   SAE_UPSCALE   20               must match the run being evaluated, or `load` fails -- or, worse,
#                                  succeeds on the wrong shapes
#   SAE_NSAMP     1600             points per level set
#
# Needs GeometricMachineLearning 0.6 or newer and HDF5. It does NOT need PoincareInvariants,
# CairoMakie or GeometricIntegrators: the invariant of a closed planar curve is its enclosed area,
# which is a shoelace sum.

using GeometricMachineLearning, HDF5, Printf
include(joinpath(@__DIR__, "structure_checks.jl"))

const weights = get(ENV, "SAE_WEIGHTS",
    joinpath(get(ENV, "GML_OUTDIR", pwd()), "pendulum_sae.h5"))
const NSAMP   = parse(Int, get(ENV, "SAE_NSAMP", "1600"))

const arch = SymplecticAutoencoder(4, 2; n_encoder_blocks=2, n_decoder_blocks=2,
    n_encoder_layers=10, n_decoder_layers=20, n_decoder_output_layers=10,
    sympnet_upscale=parse(Int, get(ENV, "SAE_UPSCALE", "20")))
const nn  = load(NeuralNetwork, weights, arch)
const enc = encoder(nn)
const dec = decoder(nn)
emb(θ, p) = embed(enc, θ, p)

# A fixed number of sample points, not a stride, so the number is comparable between runs whatever
# SAE_NSAMP is set to.
sub(v) = v[round.(Int, range(1, length(v); length=min(200, length(v))))]
function relerr(θ, p)
    num = 0.0; den = 0.0
    for (a,b) in zip(θ, p)
        x = lift(a,b); r = dec(enc(x)); num += sum(abs2, r .- x); den += sum(abs2, x)
    end
    sqrt(num/den)
end
# --- the separatrix, which every limit is read against -------------------------------------------
gm = emb(sepbranch(π/2,  3π/2, NSAMP)...)
gp = emb(sepbranch(3π/2, 5π/2, NSAMP)...)
Am, Ap = signed_area(gm), signed_area(gp)

println("─"^100)
@printf("weights  %s\n", weights)
@printf("arch     sympnet_upscale = %s, %d parameters\n",
        get(ENV, "SAE_UPSCALE", "20"), parameterlength(nn))
println("─"^100)
@printf("gamma_-  signed area %+9.4f        gamma_+  signed area %+9.4f\n", Am, Ap)
@printf("  librating limit  |gm| - |gp| = %8.4f   against a true 16\n", abs(Am) - abs(Ap))
@printf("  rotating limits  p<0 -> %8.4f   p>0 -> %8.4f   against a true 8 on both\n",
        abs(Am), abs(Ap))
println("  Opposite signs are the structural claim: the librating loop leaves gamma_+ out and the")
println("  rotating loop takes it in. Same sign means the layout has changed -- read the note before")
println("  reading anything below.")

"Print a family's table: the checks of `family_checks` with the action, its gap and the fit."
function report(title, Hs, curve)
    println()
    println(title)
    println("     H    J_angular    J_latent      gap     recon    simple   nested in the level below")
    println("  " * "─"^94)
    rows, firstbad = family_checks(enc, Hs, curve)
    for (; H, θ, p, A, sc, frac) in rows
        Ja = angular(H)
        @printf("%7.3f %10.4f  %+10.4f  %7.1f%%  %6.1f%%     %-4s    %s\n",
                H, Ja, A, 100abs(abs(A)-Ja)/Ja, 100relerr(sub(θ), sub(p)),
                sc == 0 ? "yes" : "NO",
                isnan(frac) ? "--" :
                    (frac in (0.0, 1.0) ? @sprintf("yes (%3.0f%% inside)", 100frac) :
                                          @sprintf("NO  (%3.0f%% inside)", 100frac)))
    end
    firstbad
end

lib = report("LIBRATING  (bounds a disk on the cylinder, so every chart must report the same action)",
             HLIB, H -> librating(H, NSAMP))
low = report("ROTATING, p_theta < 0  (the branch the paper's grid contains)",
             HROT, H -> rotating(H, NSAMP; sgn=-1))
upp = report("ROTATING, p_theta > 0  (absent from the paper's grid; present if SAE_FRACS=both)",
             HROT, H -> rotating(H, NSAMP; sgn=+1))

verdict(name, d) = begin
    bad = [(k, v) for (k, v) in d if v !== nothing]
    if isempty(bad)
        @printf("  %-22s sign-constant, nested and simple at every level -- consistent with an embedding\n", name)
    else
        @printf("  %-22s NOT an embedding: %s\n", name,
                join([@sprintf("%s fails from H = %.3f", k, v) for (k, v) in sort(bad, by=x->x[2])], ", "))
    end
end
println()
println("─"^100)
println("structural verdict")
verdict("librating", lib); verdict("rotating p<0", low); verdict("rotating p>0", upp)
println()
println("  `sign` is the enclosed area changing sign within a family, which a Jordan curve traversed")
println("  once cannot do. A family that SHRINKS is not a violation: one of the two rotating")
println("  families is required to nest inward. Read the three checks, not the direction.")
println()
println("  And on whichever family nests inward, the enclosed area is bounded by the inner")
println("  separatrix loop while the action grows like 2 pi sqrt(2H). Its reconstruction is")
println("  therefore bounded away from the truth for any network that embeds the cylinder, at any")
println("  capacity and on any data -- so a gap there is not something to spend GPU hours on.")
println("─"^100)
