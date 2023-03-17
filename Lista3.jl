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

# Chebychev function
    # function chebyshev(j,x)
    #     return cos(j*acos(x))
    # end

# Find roots
    # function chebyshev_root(d)

    #     roots = zeros(d+1)
    #     kgrid_roots = zeros(d+1)
            
    #     for i in 1:(d+1)
    #         roots[i] = -cos((2*i-1)/(2*(d+1)) * pi)
    #         kgrid_roots[i] = ((1 + roots[i])/2)*(k_grid[length(k_grid)]-k_grid[1]) + k_grid[1]
    #     end

    #     return roots, kgrid_roots

    # end

# Function for approximating the consumption policy function, for a given level of capital, using the Chebychev polinomial of order d
    # function c_hat(γ, k, d)
        
    #     # Transforming k to the resized grid, domain = [-1,1]
    #     k_resized = 2*(k - k_grid[1])/(k_grid[length(k_grid)] - k_grid[1]) - 1

    #     # Variable to sum for each power
    #     sum = 0
        
    #     # Loop to sum for each power
    #     for i = 1:(d+1)
    #         sum = sum + γ[i]*chebyshev(i-1, k_resized)
    #     end
        
    #     return sum
        
    # end

# Residual function

    # function R(γ, k, d, z)

    #     C0 = c_hat(γ[z,:], k, d)
    #     K1 = z_grid[z]*(k^α) + (1-δ)*k - C0

    #     one = zeros(m)
    #     two = zeros(m)
        
    #     for i = 1:m
        
    #         C1 = c_hat(γ[i,:], K1, d)
            
    #         one[i] = (1 - δ + α*z_grid[i]*K1^(α-1))
    #         two[i] = u_c(C1)
        
    #     end
        
    #     three = one .* two
    #     four = β * dot(P[z,:],three)

    #     return u_c(C0) - four
        
    # end

# Create the system of d+1 equations to solve for γ, each equation corresponds to the residual function, evaluated at a root of the chebyshev polinomial
    # function system(γ, d)

    #     aux = zeros(m, d+1)
    #     roots, k_roots = chebyshev_root(d)
        
    #     for i = 1:m
    #         for j = 1:(d+1)
    #             aux[i,j] = R(γ, k_roots[j], d, i)
    #         end
    #     end

    #     return permutedims(aux)
    
    # end

# Much faster way (and working)

S = z_grid
    
function chebyshev_root(d)
    return -cos.((2*LinRange(1,d+1,d+1) .- 1)*pi/(2*(d+1)))
end

function chebyshev(d; x)
    t = zeros(length(x),d+1)
    for j in 0:d
        t[:,j + 1] = cos.(j*acos.(x))
    end
    return t
end

function consumo(gamas;teis)
    c = teis*gamas
    return c
end


#################################
## tudo isso na função residuo ##
#################################
grid_z = z_grid
d = 1

function cb_ss(x)
    return (k_grid[length(k_grid)]-k_grid[1]).*(x .+ 1)./2 .+ k_grid[1]
end

function cb_zero(x)
    return 2 .* (x .- k_grid[1]) ./ (k_grid[length(k_grid)] - k_grid[1]) .- 1
end

function res(gamas,d,grid_z,P)
    nz = length(grid_z)
    res = zeros(d+1,nz)

    r_1 = chebyshev_root(d) # raízes para calcular o polinomio
    k_0 = cb_ss(r_1) # capital em nível

    t = chebyshev(d; x = r_1) # termos do polinomio

    c_0 = consumo(gamas; teis = t) # consumo com polinomios

    k_1 = k_0.^(1/3)*grid_z' .+ (1 - 0.012).*k_0 .- c_0
    capital_1 = cb_zero(k_1)  # normalizamos p/ [-1,1]

    for estado in 1:nz
        teis_1 = chebyshev(d; x = capital_1[:,estado]) # polinomios novos
        c_1 = consumo(gamas;teis = teis_1)
        ulinha = c_1.^(-2)
        dcdk = (1/3)*k_1[:,estado].^(-2/3)*grid_z' .+ (1-0.012) 
        lde = ulinha.*dcdk
        for id in 1:d+1
            res[id,estado] = c_0[id,estado]^(-2) - 0.987*dot(P[estado,:],lde[id,:])
        end
    end

    return res
end

res(ones(2,7),1,grid_z,P)
global guess = ones(2,7)
global s=1 
global gamaotimo = ones(2,7)
@time while s <= 5
    g(gamas) = res(gamas,s,S,P)
    gamaotimo = nlsolve(g,guess).zero
    global guess = vcat(gamaotimo, zeros(1,7))
    global s = s+1
end
gamaotimo

# Consumption policy function using the "optimal" γ

    function c_hat_single_value(γ, k, d)
            
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

    # Chebychev function
    function chebyshev_single_value(j,x)
        return cos(j*acos(x))
    end

    C = zeros(m, N)
    for i = 1:m
        for j = 1:N
            C[i,j] = c_hat_single_value(gamaotimo[:,i], k_grid[j], 5)
        end
    end

    display("image/png", plot(k_grid, 
                         permutedims(C), 
                         title="Consumption Policy Function", 
                         label=permutedims(["z = $(i)" for i in 1:m]), 
                         xlabel="Capital", 
                         ylabel="Policy (consumption)"))