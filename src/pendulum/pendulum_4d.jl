using GeometricProblems.Pendulum
using GeometricIntegrators
using GeometricMachineLearning
using CairoMakie
using Random: seed!
import GeometricProblems.Pendulum.q₀ as _q₀
import GeometricProblems.Pendulum.p₀ as _p₀
using ForwardDiff

seed!(123)

const timestep = 0.1

q₀ = _q₀
p₀ = _p₀
const l = 1.
const m = 1.
sol = integrate(podeproblem(q₀, p₀; timespan=(0.0, 30.0), timestep=timestep), ImplicitMidpoint())
q_vec = vcat(sol.dataser.q.d.parent...)
p_vec = vcat(sol.dataser.p.d.parent...)
v_vec = p_vec ./ m
q_vec_big = l .* hcat(sin.(q_vec), cos.(q_vec))'
v_vec_big = l .* hcat(cos.(q_vec), -sin.(q_vec))' .* p_vec'
p_vec_big = m .* v_vec_big
function h(q::Matrix, p::Matrix)
    p_part = sum(p .^ 2 ./ 2, dims=1) |> vec
    q_part = m .* q[2,:]
    q_part + p_part
end
dl = DataLoader((q = q_vec_big, p = p_vec_big))
n_epochs = 100000

# SympNet
arch = GSympNet(4; n_layers=1)
nn1 = NeuralNetwork(arch)
o = Optimizer(AdamOptimizer(), nn1)
batch = Batch(10)
loss1 = o(nn1, dl, batch, n_epochs)
println("Parameters: ", parameterlength(nn1))

# ResNet
arch = ResNet(4, 0, tanh)
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

ics = (q=l.*[sin(q₀[1]),cos(q₀[1])], p=l.*[cos(q₀[1]),-sin(q₀[1])]*p₀[1])
timesteps = length(sol.t)
ax2 = Axis(fig[2, 1], xlabel=L"$q$", ylabel=L"$p$", title=L"$t$=%$((timesteps - 1)*timestep)")
lines!(ax2, h(q_vec_big, p_vec_big), label="True")
pred1 = iterate(nn1, ics, n_points=timesteps)
lines!(ax2, h(pred1.q, pred1.p); label="SympNet", color=:orange)
pred2 = iterate(nn2, ics, n_points=timesteps)
lines!(ax2, h(pred2.q, pred2.p); label="ResNet", color=:purple)
axislegend(ax2)
display(fig)

save("pendulum_training.png", fig)

for i in 1:timesteps*10
    fig2 = Figure()
    ax2 = Axis(fig2[1, 1], xlabel=L"$t$", ylabel=L"$H$", title=L"$t$=%$((i - 1)*timestep)")
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
