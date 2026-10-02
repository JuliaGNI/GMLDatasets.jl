# The reference phase portrait beside the trained encoder's latent space
#
# Figure 6 of the symplectic-autoencoder paper. Both dynamical regimes, libration about theta = pi
# and rotation, with the separatrix that divides them.
#
# The rotating branch drawn is the p_theta < 0 one, which is the branch the training set contains.
# That is not cosmetic: drawn on the other branch the same weights give a picture in which the
# rotating curves do not enclose the chart at all but collapse into the small separatrix lobe and
# shrink. Which branch a latent figure draws has to be stated before any nesting is read off it.
#
# Level sets are drawn analytically rather than integrated: the encoder is a pointwise map, so the
# latent curve is exactly the image of the level set. That avoids integrator error and the
# wrap-around artefact of drawing a rotating orbit as one polyline in theta mod 2*pi.
#
# Run from the repository root:
#
#   julia --project=scripts scripts/pendulum/latent_plot.jl
#   
#   SAE_WEIGHTS=/path/to/weights.h5 HMIN=-0.4 HMAX=1.4 \
#     julia --project=scripts scripts/pendulum/latent_plot.jl
#
# Produces:
#   plots/pendulum_sae_latent.png   (OUTFILE)
#   a table of action, latent invariant, reconstruction and symplecticity per level
#
# Moved here from the symplectic-autoencoder talk's working directory
# (SciCade26/simulation_results_for_talk/pendulum_sae_latent_plot_paper.jl), where it sat beside the figure it makes.
# A figure is a claim and a generator nobody's CI runs rots against the API it was written
# for -- which this one had: see the note at the end of this header.

using GeometricMachineLearning, PoincareInvariants, HDF5, CairoMakie, Printf

const l        = 1.0
# Outputs go to GML_OUTDIR, or to the working directory. Never beside the script: a figure is a
# build product and the generator is the artefact, which is the whole reason these live here now.
const outdir  = get(ENV, "GML_OUTDIR", pwd())
const plotdir = mkpath(joinpath(outdir, "plots"))
const sae_path = get(ENV, "SAE_WEIGHTS", joinpath(outdir, "pendulum_sae.h5"))
const outfile  = get(ENV, "OUTFILE", joinpath(plotdir, "pendulum_sae_latent.png"))

# Fifteen levels straddling the separatrix H = 1: eleven librating, four rotating.
const H_levels = collect(range(parse(Float64,get(ENV,"HMIN","-0.4")), parse(Float64,get(ENV,"HMAX","1.4")); length=15))
const H_sep    = 1.0
const NPLOT    = 1500
const NQUAD    = 16_000

const arch = SymplecticAutoencoder(4, 2; n_encoder_blocks=2, n_decoder_blocks=2,
    n_encoder_layers=10, n_decoder_layers=20, n_decoder_output_layers=10, sympnet_upscale=20)
const nn  = load(NeuralNetwork, sae_path, arch)
const enc = encoder(nn)
const dec = decoder(nn)

lift(θ, pθ) = Float32[l*sin(θ), l*cos(θ), cos(θ)*pθ/l, -sin(θ)*pθ/l]

"A level set of H, sampled uniformly in its own angle; rotating orbits take the p<0 branch."
function level_set(H; n=NPLOT)
    s = 2π .* range(0, 1; length=n+1)[1:end-1]          # matches PoincareInvariants' sampling
    if H < 1
        k = sqrt((1 + H)/2)
        (π .+ 2 .* asin.(clamp.(k .* sin.(s), -1, 1)), 2 .* k .* cos.(s), 2 .* k .* cos.(s) ./ sqrt.(max.(1 .- (k .* sin.(s)).^2, eps())))
    else
        (collect(s), -sqrt.(2 .* (H .- cos.(s))), fill(1.0, n))
    end
end

encode(θs, pθs) = [enc(lift(a,b)) for (a,b) in zip(θs,pθs)]
pinv2(Z) = compute!(CanonicalFirstPI{Float64,2}(length(Z)), Matrix(reduce(hcat,[Float64.(z) for z in Z])'))
pinv4(X) = compute!(CanonicalFirstPI{Float64,4}(length(X)), Matrix(reduce(hcat,[Float64.(x) for x in X])'))

function relerr(θs, pθs)
    num = 0.0; den = 0.0
    for (a,b) in zip(θs,pθs); x = lift(a,b); r = dec(enc(x))
        num += sum(abs2, r .- x); den += sum(abs2, x) end
    sqrt(num/den)
end

println("\n     H   regime    action ∮pθdθ   latent ∮p dq   action gap   recon (rel L2)   sympl.")
println("  " * "─"^88)
lib=Float64[]; rot=Float64[]; gl=Float64[]; gr=Float64[]; sy=Float64[]
for H in H_levels
    θs, pθs, dθ = level_set(H; n=NQUAD)
    action = abs(sum(pθs .* dθ) * (2π/length(pθs)))
    Z = encode(θs, pθs); latent = abs(pinv2(Z)); dec_i = abs(pinv4([dec(z) for z in Z]))
    gap = abs(latent-action)/action; rec = relerr(θs[1:8:end], pθs[1:8:end])
    s   = abs(latent-dec_i)/latent; push!(sy,s)
    (H<1 ? (push!(lib,rec); push!(gl,gap)) : (push!(rot,rec); push!(gr,gap)))
    @printf("  %5.2f  %s  %12.5f  %13.5f   %8.1f%%   %11.1f%%   %8.1e\n",
            H, H<1 ? "librate" : "rotate ", action, latent, 100gap, 100rec, s)
end
println("  " * "─"^88)
@printf("  recon:  librating %.1f%%-%.1f%%   rotating %.1f%%-%.1f%%\n",
        100minimum(lib),100maximum(lib),100minimum(rot),100maximum(rot))
@printf("  action gap: librating %.1f%%-%.1f%%   rotating %.1f%%-%.1f%%\n",
        100minimum(gl),100maximum(gl),100minimum(gr),100maximum(gr))
@printf("  symplecticity defect: max %.1e at N = %d\n\n", maximum(sy), NQUAD)

let fig = Figure(size=(1220, 480), fontsize=25)
    # No axis titles: the paper's caption says what each panel shows (left: angular coordinates,
    # right: the learned latent space).
    ax1 = Axis(fig[1,1]; xlabel=L"\theta \;\; \mathrm{(rad)}", ylabel=L"p_\theta",
        xticks=([0,π/2,π,3π/2,2π],[L"0",L"\pi/2",L"\pi",L"3\pi/2",L"2\pi"]))
    ax2 = Axis(fig[1,2]; xlabel=L"z_q", ylabel=L"z_p")
    for ax in (ax1,ax2)
        ax.xgridcolor=(:black,0.06); ax.ygridcolor=(:black,0.06)
        ax.topspinevisible=false; ax.rightspinevisible=false
    end
    cr = (minimum(H_levels), maximum(H_levels))
    let (θs,pθs,_) = level_set(H_sep - 1e-9)      # separatrix: the H→1 limit of the librating family
        lines!(ax1, θs, pθs; color=(:black,0.6), linewidth=2.4, linestyle=:dash)
        Z = encode(θs,pθs)
        lines!(ax2, [z[1] for z in Z], [z[2] for z in Z]; color=(:black,0.6), linewidth=2.4, linestyle=:dash)
    end
    for H in H_levels
        θs,pθs,_ = level_set(H)
        lines!(ax1, θs, pθs; color=H, colormap=:viridis, colorrange=cr, linewidth=2.6)
        Z = encode(θs,pθs)
        lines!(ax2, [z[1] for z in Z], [z[2] for z in Z]; color=H, colormap=:viridis, colorrange=cr, linewidth=2.6)
    end
    Colorbar(fig[1,3]; colormap=:viridis, colorrange=cr, label=L"H",
             ticks=[-0.4,0.0,0.5,1.0,1.4], labelsize=25)
    colgap!(fig.layout,1,55); colgap!(fig.layout,2,12)
    CairoMakie.save(outfile, fig; px_per_unit=2)
    println("Saved → $outfile")
end
