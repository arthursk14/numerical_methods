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

iter = 0
for i = 1:max_iter

    RHS = u0 + β * kron(P,ones(N,1)) * reshape(v,N,m)'
    aux = mapslices(findmax, RHS, dims=2)

    g = a[[x[2] for x in aux]]
    Tv = [x[1] for x in aux]

    dist = norm(Tv-v, Inf)
    println("iter = $iter; dist = $dist")

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
                         
    M = zeros(N*m, N)

    cont = 1
    for a in 1:N
        for z in 1:m
            M[cont,findfirst(==(A_final[a,z]), a_grid)] = 1.0
            cont = cont + 1
        end
    end

    MM = zeros(N*m, N*m)

    cont = 1
    for i in 1:(m*N)
        MM[i] = kron(M[i],P[cont])
        if cont < 9
            cont = cont + 1
        else
            cont = 1
        end
    end
            
    EigenValue, EigenVector = eigen(permutedims(MM))
    LambdaVector = EigenVector[:,1]/sum(EigenVector[:,1])
    LambdaInvAV = reshape(real(LambdaVector), (N,m))

    display("image/png", plot(a_grid, 
                                  LambdaInvAV, 
                                  title="Invariant Distribution", 
                                  label=permutedims(["z = $(i)" for i in 1:m]), 
                                  xlabel="Assets", 
                                  ylabel="Density"))