using Distributions, LinearAlgebra, Plots, Random, Statistics, BenchmarkTools, Interpolations, Roots, Dates

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

# Discretizing the TFP process using Tauchen method

    # Standard deviation of z
    stdZ = σ/sqrt((1-ρ^2))

    # Lower and upper bounds of the grid
    lb = -s*stdZ                                 # Lower bound
    ub = s*stdZ                                  # Upper bound

    # Defining the n-1 possible states
    Z = LinRange(lb,ub,m)

    # Transition matrix
    z = (ub-lb)/(m-1)                            # Width between gridpoints 
    P = zeros((m, m))                            # The matrix

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
                    c = (1 - δ)*k + z*(k^α) - k_grid[h]
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

        Vi = zeros(m,N)
        Ki = zeros(m,N)

        while (dist > tol) && (iter < max_iter) 
            Vi, Ki = Iter_V(V0)
            dist = norm(Vi-V0, Inf)
            V0 = Vi
            iter = iter + 1
        end

        return Vi, Ki, iter, dist
    end

    V_final = zeros(m,N)
    K_final = zeros(m,N)

    @time V_final, K_final, iter, dist = convergence(zeros(m,N))

# Get consumption policy function

    C_final = zeros(m,N)

    for (i, z) in enumerate(z_grid)
        for (j, k) in enumerate(k_grid)
            C_final[i,j] = (1 - δ)*k + z*k^α - K_final[i,j]
        end
    end     
    
# Euler equation errors function

    # Derivative of the utility function
        function u_c(c)
            return c.^(-μ)
        end
    # Inverse of the derivative of the utility function
        function u_c_inv(c)
            return c.^(-1/μ)
        end

    function EEE(C, K, kgrid, zgrid)

        EEE = zeros(m,N)

        for (i,z) in enumerate(zgrid)
            for (j,k) in enumerate(kgrid)
                # m-vector with all possible values for u_c(c_{t+1}), that is, for all possible entries z_{t+1}
                one = u_c(zgrid*(K[i,j]^α) .+ (1-δ)*K[i,j] .- K[:,findfirst(k -> k == K[i,j], kgrid)])
                # m-vector with all possible values for (1-δ + αz_{t+1}k_{t+1}^{α-1}), that is, for all possible entries z_{t+1}
                two = (1-δ) .+ α*zgrid*(K[i,j]^(α-1))
                # Element-wise multiplication
                three = one.*two                
                # Compute the expectation on z_{t+1}, given z_t and
                four = dot(P[i,:],three)
                # Euler Equation Error
                EEE[i,j] = log10(abs(1 - u_c_inv(β*four)/C[i,j]))
            end
        end  
        
        return EEE

    end

# Compute EEE

    EEE_final = EEE(C_final,K_final,k_grid,z_grid)

# Plots

    display("image/png", plot(k_grid, 
                         permutedims(V_final), 
                         title="Value Function", 
                         label=permutedims(["z = $(i)" for i in 1:m]), 
                         xlabel="Capital", 
                         ylabel="Value"))

    display("image/png", plot(k_grid, 
                         permutedims(K_final), 
                         title="Policy Function", 
                         label=permutedims(["z = $(i)" for i in 1:m]), 
                         xlabel="Capital", 
                         ylabel="Policy (capital)"))   

    display("image/png", plot(k_grid, 
                         permutedims(C_final), 
                         title="Policy Function", 
                         label=permutedims(["z = $(i)" for i in 1:m]), 
                         xlabel="Capital", 
                         ylabel="Policy (consumption)"))  
                         
    display("image/png", plot(k_grid, 
                         permutedims(EEE_final), 
                         title="Euler Equation Errors", 
                         label=permutedims(["z = $(i)" for i in 1:m]), 
                         xlabel="Capital", 
                         ylabel="EEE"))  


# Value function iteration without maximizing (for use in the accelerator)

    function Iter_V_a(V0, K0)

        # Object to save the next value function
        V = zeros(m,N)
        
        # Choose the current z
        for (i, z) in enumerate(z_grid)

            # Choose the current k
            for (j, k) in enumerate(k_grid)

                # Calculate the consumption for that level of kprime      
                c = (1 - δ)*k + z*(k^α) - K0[i,j]

                # Calculate the whole right hand side of the Bellman equation (without the max operator) 
                V[i,j] = u(c, μ) + β * dot(P[i,:], V0[:, findfirst(k -> k == K0[i,j], k_grid)])

            end

        end

        return V

    end

# Iterations, with accelerator

    function convergence_a(V0)

        dist = Inf
        tol = 1e-5
        iter = 0
        max_iter = 1e3

        Vi = zeros(m,N)
        Ki = zeros(m,N)

        while (dist > tol) && (iter < max_iter) 
            # Run the function with optimization only once every 10 iterations, or in the first 100
            if rem(iter,10) == 0 || iter < 50 
                Vi, Ki = Iter_V(V0)
                dist = norm(Vi-V0, Inf)
                V0 = Vi
                iter = iter + 1
            else
                Vi = Iter_V_a(V0,Ki)
                dist = norm(Vi-V0, Inf)
                V0 = Vi
                iter = iter + 1
            end
        end

        return Vi, Ki, iter, dist
    end

    V_final_a = zeros(m,N)
    K_final_a = zeros(m,N)

    @time V_final_a, K_final_a, iter, dist = convergence_a(zeros(m,N))

# Get consumption policy function

    C_final_a = zeros(m,N)

    for (i, z) in enumerate(z_grid)
        for (j, k) in enumerate(k_grid)
            C_final_a[i,j] = (1 - δ)*k + z*k^α - K_final_a[i,j]
        end
    end     

# Compute EEE

    EEE_final_a = EEE(C_final_a,K_final_a,k_grid,z_grid)

# Plots

    display("image/png", plot(k_grid, 
                         permutedims(V_final_a), 
                         title="Value Function (Accelerator)", 
                         label=permutedims(["z = $(i)" for i in 1:m]), 
                         xlabel="Capital", 
                         ylabel="Value"))

    display("image/png", plot(k_grid, 
                         permutedims(K_final_a), 
                         title="Policy Function (Accelerator)", 
                         label=permutedims(["z = $(i)" for i in 1:m]), 
                         xlabel="Capital", 
                         ylabel="Policy (capital)"))   

    display("image/png", plot(k_grid, 
                         permutedims(C_final_a), 
                         title="Policy Function (Accelerator)", 
                         label=permutedims(["z = $(i)" for i in 1:m]), 
                         xlabel="Capital", 
                         ylabel="Policy (consumption)"))    
                        
    display("image/png", plot(k_grid, 
                         permutedims(EEE_final_a), 
                         title="Euler Equation Errors (Accelerator)", 
                         label=permutedims(["z = $(i)" for i in 1:m]), 
                         xlabel="Capital", 
                         ylabel="EEE"))  

# Multigrid (hard-coded for 3 grids)

    # First grid

        # Define grid
            N_1 = 100
            N = N_1
            k_grid = range(k_min, k_max, length=N_1)

        # Value Function Iteration
           @time V_1, K_1, iter, dist = convergence(zeros(m,N))

        # Get consumption policy function
            C_1 = zeros(m,N_1)

            for (i, z) in enumerate(z_grid)
                for (j, k) in enumerate(k_grid)
                    C_1[i,j] = (1 - δ)*k + z*k^α - K_1[i,j]
                end
            end     

        # Compute EEE
            EEE_1 = EEE(C_1,K_1,k_grid,z_grid)

    # Second grid

        # Define grid
            N_2 = 500
            N = N_2
            k_grid = range(k_min, k_max, length=N_2)  
        
        # Increase dimension of the previous value function by interpolating                   
            V_2 = zeros(m,N_2)

            # For each row i, interpolate the j values, using the N_1 values of the previous grid to create a N_2 grid
            for i = 1:m
                litp = linear_interpolation(range(1,N_1), V_1[i,:])
                h = 0
                for j = 1:N_2
                    # For the frist N_2/N_1 observations of V_2[i,:] I use V_1[i,1]
                    V_2[i,j] = litp(max(h+rem(j,(N_2/N_1))/(N_2/N_1),1))
                    if rem(j,(N_2/N_1)) == 0
                        h = h + 1
                    end
                end
            end

        # Value Function Iteration
            @time V_2, K_2, iter, dist = convergence(V_2)

        # Get consumption policy function
            C_2 = zeros(m,N_2)

            for (i, z) in enumerate(z_grid)
                for (j, k) in enumerate(k_grid)
                    C_2[i,j] = (1 - δ)*k + z*k^α - K_2[i,j]
                end
            end     

        # Compute EEE
            EEE_2 = EEE(C_2,K_2,k_grid,z_grid)

    # Third grid

        # Define grid
            N_3 = 5000
            N = N_3 
            k_grid = range(k_min, k_max, length=N_3)  
        
        # Increase dimension of the previous value function by interpolating                   
            V_3 = zeros(m,N_3)

            # For each row i, interpolate the j values, using the N_1 values of the previous grid to create a N_2 grid
            for i = 1:m
                litp = linear_interpolation(range(1,N_2), V_2[i,:])
                h = 0
                for j = 1:N_3
                    # For the frist N_3/N_2 observations of V_3[i,:] I use V_2[i,1]
                    V_3[i,j] = litp(max(h+rem(j,(N_3/N_2))/(N_3/N_2),1))
                    if rem(j,(N_3/N_2)) == 0
                        h = h + 1
                    end
                end
            end

        # Value Function Iteration
            @time V_3, K_3, iter, dist = convergence(V_3)

        # Get consumption policy function
            C_3 = zeros(m,N_3)

            for (i, z) in enumerate(z_grid)
                for (j, k) in enumerate(k_grid)
                    C_3[i,j] = (1 - δ)*k + z*k^α - K_3[i,j]
                end
            end     

        # Compute EEE
            EEE_3 = EEE(C_3,K_3,k_grid,z_grid)

    # Plots

    display("image/png", plot(k_grid, 
                         permutedims(V_3), 
                         title="Value Function (Multigrid)", 
                         label=permutedims(["z = $(i)" for i in 1:m]), 
                         xlabel="Capital", 
                         ylabel="Value"))

    display("image/png", plot(k_grid, 
                         permutedims(K_3), 
                         title="Policy Function (Multigrid)", 
                         label=permutedims(["z = $(i)" for i in 1:m]), 
                         xlabel="Capital", 
                         ylabel="Policy (capital)"))   

    display("image/png", plot(k_grid, 
                         permutedims(C_3), 
                         title="Policy Function (Multigrid)", 
                         label=permutedims(["z = $(i)" for i in 1:m]), 
                         xlabel="Capital", 
                         ylabel="Policy (consumption)"))    

    display("image/png", plot(k_grid, 
                         permutedims(EEE_3), 
                         title="Euler Equation Errors (Multigrid)", 
                         label=permutedims(["z = $(i)" for i in 1:m]), 
                         xlabel="Capital", 
                         ylabel="EEE"))  

# Endogenous Grid Method (EGM)

    # Bisection method function (faster than Roots.find_zero(, , Bisection()))        
        function bisection(f, a_, b_, atol = 1e-5; increasing = sign(f(b_)))
            a_, b_ = minmax(a_, b_)
            c = middle(a_,b_)
            z = f(c) * increasing
            if z > 0 #
                b = c
                a = typeof(b)(a_)
            else
                a = c
                b = typeof(a)(b_)
            end
            while abs(a - b) > atol
                c = middle(a,b)
                if f(c) * increasing > 0 #
                    b = c
                else
                    a = c
                end
            end
            a
        end

    # Iterations control parameters
        dist = Inf
        tol = 1e-5
        iter = 0
        max_iter = 1e3

    # Define again the (exogenous) grid 
        N = 500
        k_grid = range(k_min, k_max, length=N)

    # Create the matrices for the consumption policy function
        C0 = zeros(m,N)
        Ci = zeros(m,N)

    # Create the matrices for the available resource in the economy 
        # Exogenous grids, for each z
        r0 = zeros(m,N)
        # Endogenous grids, for each z
            ri = zeros(m,N)
    
    # Create the matrix for the endogenous grid
        k_grid_e = zeros(N,1)
    # Create the matrix for the policy function 
        kp_grid = zeros(N,1)

    # Initial guess for the consumption policy function (C0)
        for (i,z) in enumerate(z_grid)
            for (j,k) in enumerate(k_grid) 
                C0[i,j] = z*(k^α) + (1-δ)*k - k;
            end
        end

    # Plot the initial guess

    display("image/png", plot(k_grid, 
                         permutedims(C0), 
                         title="Consumption policy function initial guess", 
                         label=permutedims(["z = $(i)" for i in 1:m]), 
                         xlabel="Capital", 
                         ylabel="Policy (consumption)"))  

# Iterations

    # Fill the matrix of the exogenous grids
    for (i,z) in enumerate(z_grid)
        for (j,k) in enumerate(k_grid) 
            r0[i,j] = z*(k^α) + (1-δ)*k;
        end
    end
            
    while (dist > tol) && (iter < max_iter)

        # Matrix to save the RHS of the Euler Equation
        RHS = zeros(m,N)

        # Loop through all possible states
        for (i,z) in enumerate(z_grid)

            # Loop through all possible kprimes
            for (j,k) in enumerate(k_grid)  
                
                # All possible values (changing z_{t+1}) for the expression within the expectation
                one = u_c(C0[:,j]).*(z_grid.*α.*(k^(α-1)).+(1-δ))
                # Take the expectation
                two = β*dot(P[i,:],one)                
                # RHS of the Euler equation
                RHS[i,j] = u_c_inv(two)
                # Value for the endogenous grid
                ri[i,j] = RHS[i,j] + k
                
            end           
            
        end
        
        # Interpolate (extrapolate if needed) to find the new consumption police function
        for (i,z) in enumerate(z_grid)
            Ci[i,:] = LinearInterpolation(ri[i,:], RHS[i,:], extrapolation_bc=Line())[r0[i,:]]
        end    
                
        dist = norm(Ci-C0, Inf)
        iter = iter + 1
        C0 = Ci;
        
        t = now()
        print("$iter: $t, $dist \n")

    end

    display("image/png", plot(k_grid, 
                        permutedims(C0), 
                        title="Final consumption policy function", 
                        label=permutedims(["z = $(i)" for i in 1:m]), 
                        xlabel="Capital", 
                        ylabel="Policy (consumption)"))

# Iterations (old)
        
while (dist > tol) && (iter < max_iter)

    # Loop through all possible states
    for (i,z) in enumerate(z_grid)

        # Loop through all possible kprimes
        for (j,k) in enumerate(k_grid)  
            
            # All possible values (changing z_{t+1}) for the expression within the expectation
            one = u_c(C0[:,j]).*(z_grid.*α.*(k^(α-1)).+(1-δ))
            # Take the expectation
            two = β*dot(P[i,:],one)                

            # two = 0
            # for w = 1:m
            #     one = u_c(C0[w,j])*(z_grid[w]*α*(k^(α-1)) + (1-δ));
            #     two = two + β*P[i,w]*one;
            # end

            # RHS of the Euler equation
            RHS = u_c_inv(two)

            # What is the k that maximizes this kprime?
            function f(x)
                (z*(x^α) + (1-δ)*x - k) - RHS
            end

            # Save the k that maximizes 
            k_grid_e[j] = bisection(f, 0, 1e3)
            
        end
        
        # Interpolate (extrapolate if needed) to find the police function of the endogenous grid
        kp_grid = LinearInterpolation(vec(k_grid_e), collect(k_grid), extrapolation_bc=Line())
        
        # Compute the update of the consumption policy function
        for (j,k) in enumerate(k_grid)
            Ci[i,j] = z*(k^α) + (1-δ)*k - kp_grid[k]
        end
        
    end
    
    dist = norm(Ci-C0, Inf)
    iter = iter + 1
    C0 = Ci;
    
    t = now()
    print("$iter: $t, $dist \n")

end

display("image/png", plot(k_grid, 
                     permutedims(C0), 
                     title="Final consumption policy function", 
                     label=permutedims(["z = $(i)" for i in 1:m]), 
                     xlabel="Capital", 
                     ylabel="Policy (consumption)"))