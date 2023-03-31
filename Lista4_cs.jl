using Distributions, LinearAlgebra, Plots, Random, Statistics, BenchmarkTools, Interpolations, Roots, Dates, NLsolve, Latexify

    function aggregate_savings(β, ρ, σ, μ, ϕ, r)

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

            # Transform to the scale of the endowment
                z_grid = exp.(Z)

        # Solving the individual problem

            # Discretizing the asset space
                a_grid = LinRange(ϕ,-ϕ,N)

            # Utility function
                function u_vector(c,mu)
                    u = similar(c)
                    for i in eachindex(c)
                        if c[i] < 1e-10
                            u[i] = -Inf
                        else
                            u[i] = (c[i]^(1-mu)-1)/(1-mu)
                        end
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

                    if dist < tol
                        break 
                    end

                    global v = copy(Tv)

                end

                A_final = reshape(g,N,m)

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
                if dist < tol
                    break
                else
                    global Lambda = copy(LambdaInv)
                end
            end

        # Calculate aggregate savings (in this economy, we are talking about liquid bonds demand)
            global savings = 0

            for (i,z) in enumerate(z_grid)
                for (j,a) in enumerate(a_grid)
                    global savings += A_final[j,i]*LambdaInv[j,i]
                end
            end

            return savings

    end

    @time aggregate_savings(β, ρ, σ, μ, ϕ, r)

# Find equilibrium interest rate
    function aggregate_savings_difference(r)
        return aggregate_savings(β, ρ, σ, μ, ϕ, r)
    end

    # Interval for looking for values
    r_min = 0.035
    r_max = 0.045

    # Find the equilibrium interest rate r
    @time equilibrium_r = fzero(r -> aggregate_savings_difference(r), r_min, r_max)
    println("Equilibrium interest rate: ", equilibrium_r)