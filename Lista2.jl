using LinearAlgebra, Statistics, Plots

# Calibration
α = 0.33
β = 0.987
δ = 0.012
σ = 0.007
ρ = 0.95

# Grid points
m = 7    # number of grid points for the TFP process
N = 500  # number of grid points for the capital stock

k_ss = ((1/β - 1 + δ)/α)^(1/(α - 1))
k_min = 0.75*k_ss
k_max = 1.25*k_ss
k_grid = range(k_min, k_max, length=N)

# Discretize TFP process using Tauchen method
z_bar = 0
σ_z = σ / sqrt(1 - ρ^2)
m_z = 3
z_grid, P = tauchen(m_z, z_bar, ρ, σ_z)
z_grid = exp.(z_grid)

# Initialize value function and policy function
V0 = ones(N, m_z)
c0 = zeros(N, m_z)
V1 = similar(V0)
c1 = similar(c0)

# Set convergence tolerance and maximum number of iterations
tol = 1e-6
max_iter = 500
dist = Inf
iter = 0

# Value function iteration
while (dist > tol) && (iter < max_iter)
    for (i, k) in enumerate(k_grid)
        for (j, z) in enumerate(z_grid)
            # Update policy function
            c1[i, j] = (1 - δ)*k + z*k^α - k_grid[argmin(abs.(V0[:, j] - β*V0[i, :]'))]
            c1[i, j] = max(c1[i, j], 1e-10)  # impose non-negativity constraint
            # Update value function
            V1[i, j] = maximum(u.(c1[i, j], z_grid) .+ β*P[j, :]'*V0)
        end
    end
    dist = maximum(abs.(V1 - V0))
    V0, V1 = V1, V0  # swap value functions for next iteration
    c0, c1 = c1, c0  # swap policy functions for next iteration
    iter += 1
end

# Plot value function and policy function
z_idx = 1  # plot for the first TFP shock
plot(k_grid, V0[:, z_idx], xlabel="Capital", ylabel="Value", label="Value Function")
plot!(k_grid, c0[:, z_idx], xlabel="Capital", ylabel="Consumption", label="Policy Function")