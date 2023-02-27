using Random, Distributions, Plots, ARCHModels

# 1. Tauchen

    # Parameters
    n = 7                                                               # Number of grid points (given in the exercise)
    m = 3                                                               # Scaling parameter (number of deviations covered)
    rho = 0.95                                                          # Persistence parameter (given in the exercise)
    sigma = 0.007                                                       # Standard deviation of epsilon (given in te exercise)

    # Standard deviation of z
    stdZ = sigma/sqrt((1-rho^2))

    # Lower and upper bounds of the grid
    lb = -m*stdZ                                                        # Lower bound
    ub = m*stdZ                                                         # Upper bound

    # Defining the n-1 possible states
    Z = LinRange(lb,ub,n)

    # Transition matrix
    z = (ub-lb)/(n-1)                                                   # Width between gridpoints 
    P = zeros((n, n))                                                   # The matrix

    # Loop for inserting values in P[i,j]
    for i in 1:n
        for j in 1:n

            up = (Z[j]-rho*Z[i]+z/2)/sigma            
            down = (Z[j]-rho*Z[i]-z/2)/sigma          

            if j==1
                P[i,j] = cdf(Normal(),up)
            elseif j==(n)                           
                P[i,j] = 1-cdf(Normal(),down)
            else
                P[i,j] = cdf(Normal(),up)-cdf(Normal(),down)
            end
        end
    end

    # Define the names for the Tauchen transition matrix and state space
    PTauchen = P
    ZTauchen = Z

    # Display the arrays
    display(PTauchen)
    display(ZTauchen)

# 2. Rouwenhorst

    # Standard deviation of z
    stdZ = sigma/sqrt((1-rho^2))

    # Value of p
    p = (1+rho)/2

    # Lower and upper bounds of the grid
    lb = -sqrt(n-1)*stdZ                                                # Lower bound
    ub = sqrt(n-1)*stdZ                                                 # Upper bound

    # Defining the n-1 possible states
    Z = LinRange(lb,ub,n)

    # Defining the recursion function for calculating P
    function recursion(m)
        
        if m == 2
            # Defining P_2
            P = [p (1-p); (1-p) p]

        else
            # Calculate the matrix of the previous iteration (P_[n-1])
            prev = recursion(m-1)

            # Build up the 4 matrices
            p1 = p*[prev zeros(m-1,1);zeros(1,m)]
            p2 = (1-p)*[zeros(m-1,1) prev; zeros(1,m)]
            p3 = p*[zeros(1,m); zeros(m-1,1) prev]
            p4 = (1-p)*[zeros(1,m); prev zeros(m-1,1)]
            
            # Find the current matrix (P_n)
            P = p1 + p2 + p3 + p4
            
            # Dividing all but the top and bottom rows, so that the conditional probabilities add up to 1
            P[2:end-1, :] ./= 2
        end

        return P

    end

    P = recursion(n)

    # Define the names for the Tauchen transition matrix and state space
    PRouwen = P
    ZRouwen = Z

    # Display the arrays
    display(PRouwen)
    display(ZRouwen)

# 3. Simulate the processes

    # Function for finding the next realization of a discrete process, given the transition matrix, the current state and a random draw
    function nextZ(P,state,draw)      
    
        for j in 1:size(P,1)                                                    # Loop through every possible new state

            lb = quantile(Normal(0,sigma),max(min(sum(P[state,1:j-1]),1),0))      # Lower bound for the draw to represent a transition to state j
            ub = quantile(Normal(0,sigma),max(min(sum(P[state,1:j]),1),0))    # Upper bound for the draw to represent a transition to state j
                    
            if lb <= draw < ub                                                  # Check if the draw is in the interval corresponding to the conditional probability of state j
                return j                                                        # If it is, then j is the new state
            end

        end

    end

    # Initial state for the discrete processes
    initState = Int((n-1)/2)                                                    # The process starts at z_0 = 0

    # Number of periods
    t = 10000

    # Define the array for the simulations
    zTauchen = zeros(t)
    zRouwen = zeros(t)
    zContinuous = zeros(t)

    # Define the array for the states
    sTauchen = [initState for i in 1:t]
    sRouwen = [initState for i in 1:t]

    # Loop
    for i in 2:t
        local draw
        draw = rand(Normal(0,sigma))                                            # Random draw from standard normal distribution
        
        sTauchen[i] = nextZ(PTauchen,sTauchen[i-1],draw)                        # Find the next state
        sRouwen[i] = nextZ(PRouwen,sRouwen[i-1],draw)                           # Find the next state

        zTauchen[i] = ZTauchen[sTauchen[i]]                                     # Save the value of the next state
        zRouwen[i] = ZRouwen[sRouwen[i]]                                        # Save the value of the next state   
        zContinuous[i] = rho*zContinuous[i-1] + draw                                                 
    end

    # Plotting the processes together

    # Tauchen
    display("image/png", plot([zContinuous, zTauchen], label=["AR(1)" "Tauchen"]))
        xlabel!("Periods")
        ylabel!("Z")

    # Rouwenhorst
    display("image/png", plot([zContinuous, zRouwen], label=["AR(1)" "Rouwenhorst"]))
        xlabel!("Periods")
        ylabel!("Z")

# 4. Estimate the AR(1) parameter rho based on simulated data
    rhoHatCont = fit(ARMA{1,0},zContinuous).meanspec.coefs[2]
    rhoHatTauchen = fit(ARMA{1,0},zTauchen).meanspec.coefs[2]
    rhoHatRouwen = fit(ARMA{1,0},zRouwen).meanspec.coefs[2]

    println(rhoHatCont, rhoHatTauchen, rhoHatRouwen)
                    