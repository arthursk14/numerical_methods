using Distributions, LinearAlgebra, Plots, Random, Statistics, BenchmarkTools, Interpolations, Roots, Dates, NLsolve

# Calibration
    α = 0.33
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

# Utility function
    function u(c,mu)
        c = float(c)
        if c < 0
            u = -Inf
        else
            u = (c^(1-mu)-1)/(1-mu)
        end
    end

# Derivative of the utility function
    function u_c(c)
        return c.^(-μ)
    end

# Inverse of the derivative of the utility function
    function u_c_inv(c)
        return c.^(-1/μ)
    end

# Chebychev function
    function chebyshev(j,x)
        return cos(j*acos(x))
    end

# Find roots
    function chebyshev_root(d)

        roots = zeros(d+1,1)
        kgrid_roots = zeros(d+1,1)
            
        for i in 1:(d+1)
            roots[i] = -cos((2*i-1)/(2*(d+1)) * pi)
            kgrid_roots[i] = ((1 + roots[i])/2)*(k_grid[length(k_grid)]-k_grid[1]) + k_grid[1]
        end

        return roots, kgrid_roots

    end

# Function for approximating the consumption policy function, for a given level of capital, using the Chebychev polinomial of order d
    function c_hat(γ, k, d)
        
        # Transforming k to the resized grid, domain = [-1,1]
        k_resized = 2*(k - k_grid[1])/(k_grid[length(k_grid)] - k_grid[1]) - 1

        # Variable to sum for each power
        sum = 0
        
        # Loop to sum for each power
        for i = 1:(d+1)
            sum = sum + γ[i]*chebyshev(i-1, k_resized)
        end
        
        return sum
        
    end

# Residual function

    function R(γ, k, d, z)

        C0 = c_hat(γ[z,:], k, d)
        K1 = z_grid[z]*(k^α) + (1-δ)*k - C0

        one = zeros(m,1)
        two = zeros(m,1)
        
        for i = 1:m
        
            C1 = c_hat(γ[i,:], K1, d)
            
            one[i] = (1 - δ + α*z_grid[i]*K1^(α-1))
            two[i] = u_c(C1/C0)
        
        end
        
        three = one .* two

        return β * dot(P[z,:],three) - 1
        
    end

# Create the system of d+1 equations to solve for γ
    function system(γ, d)

        aux = zeros(m, d+1)
        roots, k_roots = chebyshev_root(d)
        
        for i = 1:m
            for j = 1:(d+1)
                aux[i,j] = R(γ, k_roots[j], d, i)
            end
        end
        
        return reshape(aux, :, 1)
    
    end


# Loop to find γ_star, starting with a guess for the polinomial of order 2; and then using the result of the current iteration as the guess for the next d
    for i = 1:d 
        if i == 1
            global γ0 = ones(m, i+1)
            local function f(γ)
                return system(γ, i)
            end
            global γ_star = nlsolve(f, γ0)
        else
            γ_new = hcat(γ_star.zero[:,1],ones(m,1))
            local function f(γ)
                return system(γ, i)
            end
            global γ_star = nlsolve(f, γ_new)
         end
    end