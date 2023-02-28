using Distributions, LinearAlgebra, Plots, Random, Statistics, BenchmarkTools

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

# Value function iteration

    function Iter_V(V0)

        # Object to save the next value function
        V = zeros(m,N)

        # Object to save the policy function
        K = zeros(m,N)

        # Choose the current z
        for (i, z) in enumerate(z_grid)

            # Monotonicity control
            mono_control = 1

            # Choose the current k
            for (j, k) in enumerate(k_grid)

                RHS_old = -Inf

                # For the given (z, k), we look for the optimal kprime, using monotonicity to start the loop
                for h in mono_control:N
                    # Calculate the consumption for that level of kprime      
                    c = (1 - δ)*k + z*k^α - k_grid[h]
                    # Calculate the whole right hand side of the Bellman equation (without the max operator) 
                    RHS = u(c, μ) + β * dot(P[i,:], V0[:, h])
                    # Here we exploit concavity, if the value declines, the previous level is a maximum
                    if RHS < RHS_old
                        # With entries (z,k), the value function returns the maximum RHS achievable
                        V[i,j] = RHS_old
                        # Updating the monotonicity control
                        mono_control = h-1
                        # The policy function is the kprime (or consumption, one-to-one) chosen to maximize the RHS
                        K[i,j] = k_grid[h-1]
                        break
                    else
                        # If we exhaust the grid search
                        if h == N
                            # Then and the maximum is the current level for the RHS
                            V[i,j] = RHS                            
                            # Updating the monotonicity control
                            mono_control = h
                            # The policy function is the kprime (or, alternatively the consumption in t) chosen to maximize the RHS
                            K[i,j] = k_grid[h]
                        else
                            # If not, this was not the maximum, so save for next iteration
                            RHS_old = RHS
                        end
                    end
                end
            end
        end

        return V, K

    end

# Iterations

    function convergence(V0)

        dist = Inf
        tol = 1e-5
        iter = 0
        max_iter = 1e3

        while (dist > tol) && (iter < max_iter) 
            Vi, Ki = Iter_V(V0)
            dist = norm(V-V0, Inf)
            V0 = V
            iter = iter + 1
        end

        return Vi, Ki, iter, dist
    end

    @btime V_final, K_final, iter, dist = convergence(zeros(m,N))

    display( "image/png", plot( k_grid, [ V_final[1,:] V_final[2,:] V_final[3,:] V_final[4,:] V_final[5,:] V_final[6,:] V_final[7,:] ], 
                                title="Value Function", 
                                label=[ "z = 1" "z = 2" "z = 3" "z = 4" "z = 5" "z = 6" "z = 7" ] ) )

    display( "image/png", plot( k_grid, [ K_final[1,:] K_final[2,:] K_final[3,:] K_final[4,:] K_final[5,:] K_final[6,:] K_final[7,:] ], 
                                title="Policy Function", 
                                label=[ "z = 1" "z = 2" "z = 3" "z = 4" "z = 5" "z = 6" "z = 7" ] ) )
                               