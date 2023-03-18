using Distributions, LinearAlgebra, Plots, Random, Statistics, BenchmarkTools, Interpolations, Roots, Dates, NLsolve

# Calibration
    α = 1/3
    β = 0.987
    δ = 0.012
    σ = 0.007
    ρ = 0.95
    μ = 2

# Grid points
    m = 7                                        # Number of grid points for the TFP process
    N = 500                                      # Number of grid points for the capital stock
    s = 3                                        # Scaling parameter for the Tauchen discretization

# Capital stock
    k_ss = ((1/β - 1 + δ)/α)^(1/(α - 1))
    k_min = 0.75*k_ss
    k_max = 1.25*k_ss
    k_grid = range(k_min, k_max, length=N)

# Order of the polinomial
    d = 5

# Discretizing the TFP process using Tauchen method

    # Standard deviation of z
        stdZ = σ/sqrt((1-ρ^2))

    # Lower and upper bounds of the grid
        lb = -s*stdZ                             # Lower bound
        ub = s*stdZ                              # Upper bound

    # Defining the n-1 possible states
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

    # Transform to the original scale of the AR(1) process
        z_grid = exp.(Z)

# Derivative of the utility function
    function u_c(c)
        return c.^(-μ)
    end

# Find roots of chebychev polinomial of order d
    function chebyshev_root(d)
        return -cos.((2*LinRange(1,d+1,d+1) .- 1)*pi/(2*(d+1)))
    end

# Find the values of the terms of a chebychev polinomial of order d
    function chebyshev(d; x)
        t = zeros(length(x),d+1)
        for j in 0:d
            t[:,j + 1] = cos.(j*acos.(x))
        end
        return t
    end

# Approximate the consumption function using the terms of the polinomial and weights γ
    function C_hat(γ; t)
        return t*γ
    end

# Functions to adapt the grid to [-1,1] and vice-versa

    function transform(x)
        return (k_grid[length(k_grid)]-k_grid[1]).*(x .+ 1)./2 .+ k_grid[1]
    end

    function transform_back(x)
        return 2 .* (x .- k_grid[1]) ./ (k_grid[length(k_grid)] - k_grid[1]) .- 1
    end

# Residual function, we create the system of d+1 equations to solve for γ, each equation corresponds to the residual function, evaluated at a root of the chebyshev polinomial
    function res(γ,d)

        res = zeros(d+1,m)
        roots = chebyshev_root(d)

        K0 = transform(roots)
        t0 = chebyshev(d; x = roots)
        C0 = C_hat(γ; t = t0)

        K1 = K0.^α*z_grid' .+ (1 - δ).*K0 .- C0
        K1_grid = transform_back(K1)

        for i in 1:m
            t1 = chebyshev(d; x = K1_grid[:,i])
            c_1 = C_hat(γ; t = t1)
            one = c_1.^(-2)
            two = α*K1[:,i].^(α-1)*z_grid' .+ (1-δ) 
            three = one.*two
            for j in 1:d+1
                res[j,i] = C0[j,i]^(-2) - β * dot(P[i,:],three[j,:])
            end
        end

        return res
    end

# Iterations to get the initial guess for the order d polinomial as the value the solves the polinomial of order d-1, starting with γ = 1 for d = 1
    global γ0 = ones(2,7)
    global h = 1 
    global γ_star = ones(2,7)

    @time while h <= d
        f(γ) = res(γ,h)
        global γ_star = nlsolve(f,γ0).zero
        global γ0 = vcat(γ_star, zeros(1,m))
        global h = h + 1
    end

# Consumption policy function using the "optimal" γ

    # Chebychev function for a scalar
        function chebyshev_scalar(j,x)
            return cos(j*acos(x))
        end

    # Function for approximating the consumption policy function, for a given level of capital, using the Chebychev polinomial of order d, with parameters γ
    function c_hat(γ, k, d)
            
        # Transforming k to the resized grid, domain = [-1,1]
        k_resized = 2*(k - k_grid[1])/(k_grid[length(k_grid)] - k_grid[1]) - 1

        # Variable to sum for each power
        sum = 0
        
        # Loop to sum for each power
        for i = 1:(d+1)
            sum = sum + γ[i]*chebyshev_scalar(i-1, k_resized)
        end
        
        return sum
        
    end

    # Recover the whole function for consumption policy, using γ_star
    C = zeros(m, N)
    for i = 1:m
        for j = 1:N
            C[i,j] = c_hat(γ_star[:,i], k_grid[j], 5)
        end
    end

    # Plot
    display("image/png", plot(k_grid, 
                         permutedims(C), 
                         title="Consumption Policy Function", 
                         label=permutedims(["z = $(i)" for i in 1:m]), 
                         xlabel="Capital", 
                         ylabel="Policy (consumption)"))

    # Recover the whole function for capital policy, using γ_star
    K = zeros(m, N)
    for (i, z) in enumerate(z_grid)
        for (j, k) in enumerate(k_grid)
            K[i,j] = (1 - δ)*k + z*k^α - C[i,j]
        end
    end

    # Plot
    display("image/png", plot(k_grid, 
                         permutedims(K), 
                         title="Capital Policy Function", 
                         label=permutedims(["z = $(i)" for i in 1:m]), 
                         xlabel="Capital", 
                         ylabel="Policy (capital)"))

# Recover value function

    # Utility function
        function u(c,mu)
            c = float(c)
            if c < 0
                u = -Inf
            else
                u = (c^(1-mu)-1)/(1-mu)
            end
        end

    # Approximate to the defined grid
        i_grid = zeros(Int,m,N)
        for (i, z) in enumerate(z_grid)
            for (j, k) in enumerate(k_grid)
                dif = abs.(K[i,j] .- k_grid)
                aux = min(dif...)
                i_grid[i,j] = trunc(Int, findfirst(a -> a == aux, dif))
            end
        end

    # Value function iteration without maximizing (the same we used for the accelerator)
        function Iter_V(V0, K0)

            # Object to save the next value function
            V = zeros(m,N)
            
            # Choose the current z
            for (i, z) in enumerate(z_grid)

                # Choose the current k
                for (j, k) in enumerate(k_grid)

                    # Calculate the consumption for that level of kprime      
                    c = (1 - δ)*k + z*(k^α) - K0[i,j]

                    # Calculate the whole right hand side of the Bellman equation (without the max operator) 
                    V[i,j] = u(c, μ) + β * dot(P[i,:], V0[:, i_grid[i,j]])

                end

            end

            return V

        end

    # Iterations

    function convergence(V0, K0)

        dist = Inf
        tol = 1e-5
        iter = 0
        max_iter = 1e3

        Vi = zeros(m,N)

        while (dist > tol) && (iter < max_iter) 
            Vi = Iter_V(V0, K0)
            dist = norm(Vi-V0, Inf)
            V0 = Vi
            iter = iter + 1
        end

        return Vi, iter, dist
    end

    @time V, iter, dist = convergence(zeros(m,N), K)
    
    # Plot
    display("image/png", plot(k_grid, 
                         permutedims(V), 
                         title="Value Function", 
                         label=permutedims(["z = $(i)" for i in 1:m]), 
                         xlabel="Capital", 
                         ylabel="Value"))