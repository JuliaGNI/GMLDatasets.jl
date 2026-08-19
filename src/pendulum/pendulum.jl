using GeometricProblems.Pendulum
using GeometricIntegrators
using GeometricMachineLearning
using CairoMakie
using Random: seed!
import GeometricProblems.Pendulum.q₀ as _q₀
import GeometricProblems.Pendulum.p₀ as _p₀
using ForwardDiff

seed!(123)

const timestep = 0.2

q₀ = _q₀
p₀ = _p₀
sol = integrate(podeproblem(q₀, p₀; timespan=(0.0, 15.0), timestep=timestep), ImplicitMidpoint())
dl = DataLoader(sol)
n_epochs = 100000

# SympNet
arch = GSympNet(2; n_layers=5)
nn1 = NeuralNetwork(arch)
o = Optimizer(AdamOptimizer(), nn1)
batch = Batch(10)
loss1 = o(nn1, dl, batch, n_epochs)
println("Parameters: ", parameterlength(nn1))

# ResNet
arch = ResNet(2, 4, tanh)
nn2 = NeuralNetwork(arch)
o = Optimizer(AdamOptimizer(), nn2)
batch = Batch(10)
loss2 = o(nn2, dl, batch, n_epochs)
println("Parameters: ", parameterlength(nn2))

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Epoch", ylabel="Loss")
lines!(ax, loss1; label=L"SympNet; $n_\mathrm{params}$=%$(parameterlength(nn1))", color=:orange)
lines!(ax, loss2; label=L"ResNet; $n_\mathrm{params}$=%$(parameterlength(nn2))", color=:purple)
axislegend(ax)
display(fig)

ics = (q=q₀, p=p₀)
timesteps = length(sol.t)
ax2 = Axis(fig[2, 1], xlabel=L"$q$", ylabel=L"$p$", title=L"$t$=%$((timesteps - 1)*timestep)")
lines!(ax2, vcat(sol.dataser.q.d...), vcat(sol.dataser.p.d...), label="True")
pred1 = iterate(nn1, ics, n_points=timesteps)
lines!(ax2, pred1.q[1, :], pred1.p[1, :]; label="SympNet", color=:orange)
pred2 = iterate(nn2, ics, n_points=timesteps)
lines!(ax2, pred2.q[1, :], pred2.p[1, :]; label="ResNet", color=:purple)
axislegend(ax2)
display(fig)

save("plots/pendulum_training.png", fig)

for i in 1:timesteps*100
    fig2 = Figure()
    ax2 = Axis(fig2[1, 1], xlabel=L"$q$", ylabel=L"$p$", title=L"$t$=%$((i - 1)*timestep)")
    pred1 = iterate(nn1, ics, n_points=i)
    pred2 = iterate(nn2, ics, n_points=i)
    if i < timesteps
        lines!(ax2, vcat(dl.input.q)[1:i], vcat(dl.input.p)[1:i], label="True")
    else
        lines!(ax2, vcat(sol.dataser.q.d...), vcat(sol.dataser.p.d...), label="True")
    end
    lines!(ax2, pred1.q[1, :], pred1.p[1, :]; label="SympNet", color=:orange)
    lines!(ax2, pred2.q[1, :], pred2.p[1, :]; label="ResNet", color=:purple)
    axislegend(ax2)
    save("Animations/sympnet_pendulum/pendulum_$(i).png", fig2)
end

# check symplecticity
f(x) = nn2(x, nn2.params)
A(x) = ForwardDiff.jacobian(f, x)
J = hcat()
