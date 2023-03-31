using Distributions, LinearAlgebra, Plots, Random, Statistics, BenchmarkTools, Interpolations, Roots, Dates, NLsolve, Latexify

# Calibration
    β = 0.96
    ρ = 0.90
    σ = 0.01
    μ = 1.0001

# Grid points
    m = 9                                        # Number of grid points for endowment shocks
    N = 500                                      # Number of grid points for the asset holding
    s = 3                                        # Scaling parameter for the Tauchen discretization

# Discretizing the stochastic process using Tauchen method

    # Standard deviation of z
        stdZ = σ/sqrt((1-ρ^2))

    # Lower and upper bounds of the grid
        lb = -s*stdZ                             # Lower bound
        ub = s*stdZ                              # Upper bound

    # Defining the m-1 possible states
        Z = LinRange(lb,ub,m)

    # Transition matrix
        z = (ub-lb)/(m-1)                        # Width between gridpoints 
        P = zeros((m, m))                        # The matrix

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

    # Transform to the scale of the endowment
        z_grid = exp.(Z)

# Solving the individual problem

    # Initial interest rate
        r = 1/β-1

    # Natural debt limit
        ϕ = lb/r

    # Discretizing the asset space
        a_grid = LinRange(ϕ,-ϕ,N)

    # Utility function
        function u(c,mu)
            c = float(c)
            if c < 1e-10
                u = -Inf
            else
                u = (c^(1-mu)-1)/(1-mu)
            end
        end

    # Value function iteration

        grid = zeros(N*m,2)
        for (i, z) in enumerate(z_grid)
            for (j, a) in enumerate(a_grid)
                grid[j+(N*(i-1)),1] = a
                grid[j+(N*(i-1)),2] = z
            end
        end     

        n = size(grid,1)

        a_vector = grid[:,1]
        z_vector = grid[:,2]

        C = zeros(n,N)
        u0 = zeros(n,N)
        for (i, a) in enumerate(a_grid)
            C[:,i] = (1 + r).*a_vector + z_vector .- a_grid[i]
            u0[:,i] = u.(C[:,i],μ)
        end

        Vguess = zeros(N,m)
        for (i,z) in enumerate(z_grid)
            for (j,a) in enumerate(a_grid)
                Vguess[j,i] = u(r*a+z, μ)/(1-β)
            end
        end

        global v = reshape(Vguess, :, 1)
        global g = similar(a_vector)        
        global dist = Inf

        max_iter = 1e3
        tol = 1e-5
        

        for i = 1:max_iter

            RHS = u0 + β * kron(P,ones(N,1)) * reshape(v,N,m)'
            aux = mapslices(findmax, RHS, dims=2)

            global g = a_vector[[x[2] for x in aux]]
            Tv = [x[1] for x in aux]

            global dist = norm(Tv-v, Inf)
            println("iter = $i; dist = $dist")

            if dist < tol
                break 
            end

            global v = copy(Tv)

        end

        c = (1 + r)*a_vector + z_vector - g

        V_final = reshape(v,N,m)
        A_final = reshape(g,N,m)
        C_final = reshape(c,N,m)

    # Plots

        display("image/png", plot(a_grid, 
                            V_final, 
                            title="Value Function", 
                            label=permutedims(["z = $(i)" for i in 1:m]), 
                            xlabel="Assets", 
                            ylabel="Value"))

        display("image/png", plot(a_grid, 
                            A_final, 
                            title="Policy Function", 
                            label=permutedims(["z = $(i)" for i in 1:m]), 
                            xlabel="Assets", 
                            ylabel="Policy (assets)")) 

        display("image/png", plot(a_grid, 
                            C_final, 
                            title="Policy Function", 
                            label=permutedims(["z = $(i)" for i in 1:m]), 
                            xlabel="Assets", 
                            ylabel="Policy (consumption)"))


# Find the invariant distribution
        global Lambda = ones(N,m)/(N*m)
        global dist = 1

        # Iterate to find the invariant distribution
        for iter in 1:max_iter
            global LambdaInv = zeros(N, m)    
            
            for (i,a) in enumerate(a_grid)
                for (j,z) in enumerate(z_grid)
                    # Find the next asset grid index
                    next_a = A_final[i, j]
                    index_next_a = findmin(abs.(A_final[:, j] .- next_a))[2]
                    
                    # Update the joint distribution
                    for k in 1:m
                        LambdaInv[index_next_a, k] += P[j, k] * Lambda[i, j]
                    end
                end
            end
            
            # Check for convergence
            global dist = norm(Lambda - LambdaInv, Inf)
            println("iter = $iter; dist = $dist")
            if dist < tol
                break
            else
                global Lambda = copy(LambdaInv)
            end
        end

        # Compute the marginal distributions
        marginal_assets = sum(LambdaInv, dims=2)[:, 1]
        marginal_shocks = sum(LambdaInv, dims=1)[1, :]

        # Plots
        display("image/png", plot(a_grid, 
                                  marginal_assets, 
                                  title="Invariate distribution of assets", 
                                  xlabel="Assets", 
                                  ylabel="Density",
                                  legend=false))

        display("image/png", plot(z_grid, 
                                  marginal_shocks, 
                                  title="Invariate distribution of income (shocks)", 
                                  xlabel="Income", 
                                  ylabel="Density",
                                  legend=false))