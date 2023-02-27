using Random, Distributions, LinearAlgebra, Statistics, Plots

# Calibration
    α = 0.33
    β = 0.987
    δ = 0.012
    σ = 0.007
    ρ = 0.95
    μ = 2
    
# Grid points
    m = 7                                                               # Number of grid points for the TFP process
    N = 500                                                             # Number of grid points for the capital stock
    s = 3                                                               # Scaling parameter for the Tauchen discretization

# Capital stock
    k_ss = ((1/β - 1 + δ)/α)^(1/(α - 1))
    k_min = 0.75*k_ss
    k_max = 1.25*k_ss
    k_grid = range(k_min, k_max, length=N)

# Discretizing the TFP process using Tauchen method

    # Standard deviation of z
    stdZ = σ/sqrt((1-ρ^2))

    # Lower and upper bounds of the grid
    lb = -s*stdZ                                                        # Lower bound
    ub = s*stdZ                                                         # Upper bound

    # Defining the n-1 possible states
    Z = LinRange(lb,ub,m)

    # Transition matrix
    z = (ub-lb)/(m-1)                                                   # Width between gridpoints 
    P = zeros((m, m))                                                   # The matrix

    # Loop for inserting values in P[i,j]
    for i in 1:m
        for j in 1:m

            up = (Z[j]-ρ*Z[i]+z/2)/σ            
            down = (Z[j]-ρ*Z[i]-z/2)/σ          

            if j==1
                P[i,j] = cdf(Normal(),up)
            elseif j==(m)                           
                P[i,j] = 1-cdf(Normal(),down)
            else
                P[i,j] = cdf(Normal(),up)-cdf(Normal(),down)
            end
        end
    end

    # Display the arrays

    display(P)
    display(Z)

    # Transform to the original scale of the AR(1) process
    z_grid = exp.(Z)

# Utility function
    function u(c,mu)
        c = float(c)
        if c <= 0
            u = -Inf
        else
            if μ == 1
                u = log(c)                
            else
                u = (c^(1-μ)-1)/(1-μ)
            end
        end
    end

# Initialize value and policy function
    V0 = zeros(m, N)
    V = zeros(m, N)

# Set control parameters for the algorithm 
    tol = 1e-5
    max_iter = 500
    dist = Inf
    iter = 0

# Value function iteration
    while (dist > tol) && (iter < max_iter)
        for (i, k) in enumerate(k_grid)
            for (j, z) in enumerate(z_grid)
                # For the given (k, z), we look for the optimal kprime
                for (h, kprime) in enumerate(k_grid)                    
                    c = (1 - δ)*k + z*k^α - kprime
                    RHS = u(c, μ) + β * dot(P[j,:], V0[:, h])   


                end

                # Update policy function
                c = (1 - δ)*k + z*k^α - k_grid[argmin(abs.(V0[:, j] - β*V0[i, :]'))]
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