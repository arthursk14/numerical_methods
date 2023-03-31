using Distributions, LinearAlgebra, Plots, Random, Statistics, BenchmarkTools, Interpolations, Roots, Dates, NLsolve, Latexify

# Calibration
    β = 0.96
    ρ = 0.9
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
        r = 1/β - 1

    # Natural debt limit
        ϕ = lb/r

    # Discretizing the asset space
        a_grid = LinRange(-ϕ,ϕ,N)

    # Utility function
        function u(c,mu)
            c = float(c)
            if c < 0
                u = -Inf
            else
                u = (c^(1-mu)-1)/(1-mu)
            end
        end

    # Value function iteration
        function Iter_V(V0)
        # function Iter_V(V0, file)

            # Object to save the next value function
            V = zeros(m,N)

            # Object to save the policy function
            A = zeros(m,N)

            # Choose the current z
            for (i, z) in enumerate(z_grid)

                # Monotonicity control
                mono_control = 1

                # Choose the current a
                for (j, a) in enumerate(a_grid)

                    RHS_old = -Inf

                    # For the given (z, a), we look for the optimal aprime, using monotonicity to start the loop
                    for h in mono_control:N
                        # Calculate the consumption for that level of aprime      
                        c = (1+r)*a + z - a_grid[h]
                        # Calculate the whole right hand side of the Bellman equation (without the max operator) 
                        RHS = u(c, μ) + β * (1+r) * dot(P[i,:], V0[:, h])
                        
                        # println(file, "iz = $i; ja = $j; h = $h; RHS = $RHS; RHS_old = $RHS_old")
                        # flush(file)

                        # Here we exploit concavity, if the value declines, the previous level is a maximum                        
                        if RHS < RHS_old
                            # With entries (z,a), the value function returns the maximum RHS achievable
                            V[i,j] = RHS_old
                            # Updating the monotonicity control
                            mono_control = h-1
                            # The policy function is the aprime (or consumption, one-to-one) chosen to maximize the RHS
                            A[i,j] = a_grid[h-1]
                            break
                        else
                            # If we exhaust the grid search
                            if h == N
                                # Then and the maximum is the current level for the RHS
                                V[i,j] = RHS                            
                                # Updating the monotonicity control
                                mono_control = h
                                # The policy function is the aprime (or, alternatively the consumption in t) chosen to maximize the RHS
                                A[i,j] = a_grid[h]
                            else
                                # If not, this was not the maximum, so save for next iteration
                                RHS_old = RHS
                            end
                        end
                    end
                end
            end

            return V, A

        end

    # Iterations
        function convergence(V0)            

            Vi = zeros(m,N)
            Ai = zeros(m,N)

            # open("log.txt", "w") do txt

                global max_iter = 1e3
                global tol = 1e-5
                global iter = 0
                global dist = Inf

                while (dist > tol) && (iter < max_iter) 
                    # Vi, Ai = Iter_V(V0, txt)
                    Vi, Ai = Iter_V(V0)
                    global dist = norm(Vi-V0, Inf)
                    V0 = copy(Vi)
                    global iter = iter + 1                            
                    # println(txt, "iter = $iter; dist = $dist")
                    # flush(txt)
                end
            
            # end

            return Vi, Ai, iter, dist
        end

        V_final = zeros(m,N)
        A_final = zeros(m,N)

        @time V_final, A_final, iter, dist = convergence(zeros(m,N))

    # Get consumption policy function
        C_final = zeros(m,N)

        for (i, z) in enumerate(z_grid)
            for (j, a) in enumerate(a_grid)
                C_final[i,j] = (1 + r)*a + z - A_final[i,j]
            end
        end     

        
    # Plots

        display("image/png", plot(a_grid, 
                                  permutedims(V_final), 
                                  title="Value Function", 
                                  label=permutedims(["z = $(i)" for i in 1:m]), 
                                  xlabel="Assets", 
                                  ylabel="Value"))

        display("image/png", plot(a_grid, 
                                  permutedims(A_final), 
                                  title="Policy Function", 
                                  label=permutedims(["z = $(i)" for i in 1:m]), 
                                  xlabel="Assets", 
                                  ylabel="Policy (assets)"))   

        display("image/png", plot(a_grid, 
                                  permutedims(C_final), 
                                  title="Policy Function", 
                                  label=permutedims(["z = $(i)" for i in 1:m]), 
                                  xlabel="Assets", 
                                  ylabel="Policy (consumption)"))

# Invariant distribution and aggregate savings

    global LambdaInv = ones(m, N) ./ (m * N)
    global Lambda = zeros(m, N)

    global iter = 0
    global tol = 1e-5
    global dist = norm(LambdaInv-Lambda, Inf)

    # Loop
    while dist > tol
        for zlin in 1:m
            for alin in 1:N
                sum = 0
                for z in 1:m
                    for a in 1:N
                        sum += (a_grid[alin] == A_final[z, a] ? 1 : 0) * LambdaInv[z,a] * P[zlin,z]
                    end
                end
                Lambda[zlin,alin] = sum
            end
        end

        global iter = iter + 1
        global dist = norm(Lambda-LambdaInv, Inf)

        print("Iteration: $iter; Dist: $dist")

        global LambdaInv = copy(Lambda)
        global Lambda = zeros(m, N)
    end

