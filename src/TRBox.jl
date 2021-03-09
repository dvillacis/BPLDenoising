########################################################
# Bilevel Parameter Learning via Nonsmooth Trust Region (Vectorized Mode)
########################################################

using SparseArrays

using AlgTools
using AlgTools.Util
import AlgTools.Iterate

using VariationalImaging.GradientOps
using VariationalImaging.Util

using LinearAlgebra
using LinearOperators,Krylov

export bilevel_learn

#############
# Data types
#############

ImageSize = Tuple{Integer,Integer}
Image = Array{Float64,2}
Primal = Image
Dual = Array{Float64,3}
Parameter = Union{Real,AbstractArray}
Dataset = Tuple{Array{Float64,3},Array{Float64,3}}

#########################
# Iterate initialisation
#########################

function init_rest(x::Real,learning_function::Function,Δ,ds)
    x̄ = copy(x)
    u,fx,gx = learning_function(x,ds,Δ)
    ū = copy(u)
    fx̄ = copy(fx)
    gx̄ = copy(gx)
    B = 0.1
    return x, x̄, u, ū, fx, gx, fx̄, gx̄, Δ, B
end

function init_rest(x::Union{AbstractArray{T,3},AbstractArray{T,2},AbstractArray{T,1}},learning_function::Function,Δ,ds) where T
    x̄ = copy(x)
    u,fx,gx = learning_function(x,ds,Δ)
    ū = copy(u)
    fx̄ = copy(fx)
    gx̄ = copy(gx)
    B = LBFGSOperator(length(x[:]))
    return x, x̄, u, ū, fx, gx, fx̄, gx̄, Δ, B
end


############
# Auxiliary Functions
############

# Dogleg step calculation using l_\infty ball
function dogleg_box(x::Real,gx,B,Δ)
    lb,ub = get_bounds(x,Δ)
    #@info lb,ub
    pn = B\gx
    if in_bounds(lb,Δ,pn)
        return pn
    end
    p = -(norm(gx)^2/(gx'*(B*gx)))*gx # Cauchy Step
    if in_bounds(lb,Δ,p) == false
        #@info "Scaled"
        t = step_to_bound(p/norm(p),lb,Δ)
        return (p/norm(p))*t
    end
    #@info "Dogleg"
    t = step_to_bound(pn-p,lb,Δ)
    return p + t * (pn-p)
end

function dogbox(x::Real,gx,B,Δ)
    lb,ub = get_bounds(x,Δ)
    #@info lb,ub
    pn = B\gx
    if in_bounds(lb,Δ,pn)
        return pn
    end
    p = -(norm(gx)^2/(gx'*(B*gx)))*gx # Cauchy Step
    if in_bounds(lb,Δ,p) == false
        @info "Dogbox"
        t = step_to_bound(p,lb,Δ)
        psc = p*t
        t2 = step_to_bound(psc-pn,lb,Δ)
        return psc + t2 * (psc-pn)
    end
    #@info "Dogleg"
    t = step_to_bound(pn-p,lb,Δ)
    return p + t * (pn-p)
end

# Dogleg step calculation using l_\infty ball
function dogleg_box(x::Union{AbstractArray{T,3},AbstractArray{T,2},AbstractArray{T,1}},gx,B,Δ) where T
    lb,ub = get_bounds(x,Δ)
    pn = newton_step(B,gx)
    if in_bounds(lb,Δ,pn)
        return pn
    end
    p = cauchy_step(B,gx)
    if in_bounds(lb,Δ,p) == false
        @info "Scaled"
        t = step_to_bound(p/norm₂(p),lb,Δ)
        return (p/norm₂(p)) .*t
    end
    @info "Dogleg"
    t = step_to_bound(pn-p,lb,Δ)
    return p + t .* (pn-p)
end

function dogbox(x::Union{AbstractArray{T,3},AbstractArray{T,2},AbstractArray{T,1}},gx,B,Δ) where T
    lb,ub = get_bounds(x,Δ)
    pn = newton_step(B,gx)
    if in_bounds(lb,Δ,pn)
        return pn
    end
    p = cauchy_step(B,gx)
    if in_bounds(lb,Δ,p) == false
        @info "Dogbox"
        t = step_to_bound(p,lb,Δ)
        psc = t .* p
        t2 = step_to_bound(pn-psc,lb,Δ)
        return psc + t2 .* (pn-psc)
    end
    @info "Dogleg"
    t = step_to_bound(pn-p,lb,Δ)
    return p + t .* (pn-p)
end

function newton_step(B::LinearOperators.LBFGSOperator,gx::AbstractArray)
    pn, ks = Krylov.cg_lanczos(B,-gx[:])
    if ks.solved == false
        @error ks
    end
    return reshape(pn,size(gx)...)
end

function cauchy_step(B::LinearOperators.LBFGSOperator,gx::AbstractArray)
    p = -(norm(gx[:])^2/(gx[:]'*(B*gx[:])))*gx[:]
    return reshape(p,size(gx)...)
end

# Distance to l_\infty bound
function step_to_bound(p,lb,ub)
    dist = max.(lb ./ p, ub ./ p)
    return dist
end

# Test if vector is within the l_\infty ball
function in_bounds(lb,ub,x)
    return all(x .>= lb) && all(x .<= ub)
end

# Get bounds for the l_\infty ball intersected with positive quadrant
function get_bounds(x,Δ)
    lb = max.(-Δ,eps() .-x)
    ub = Δ * ones(size(x))
    return lb,ub
end

function pred(B::LinearOperators.LBFGSOperator,p,gx)
    return -p[:]'*gx[:] -0.5 * p[:]'*(B*p[:])
end

function pred(B::Real,p,gx)
    return -p*gx - 0.5*p*B*p
end

function updateBFGS!(B::LinearOperators.LBFGSOperator,y,s)
    if y[:]'*(B*y[:]) > 0
        push!(B,y[:],s[:])
    end
    return B
end

function updateBFGS!(B::Real,y,s)
    if y'*(B*y) > 0
        B = B + (y*y)/(y*s) - (B*s*s*B)/(s*B*s)
    end
    return B
end

############
# Algorithm
############

function bilevel_learn(ds :: Dataset,
    learning_function::Function;
    xinit :: Parameter,
    iterate = Iterate.simple_iterate,
    params::NamedTuple)

    ################################                                        
    # Extract and set up parameters
    ################################                    

    η₁, η₂ = params.η₁, params.η₂
    β₁, β₂ =  params.β₁, params.β₂
    Δ₀ = params.Δ₀
    parsize = size(xinit)

    ######################
    # Initialise iterates
    ######################
    x, x̄, u, ū, fx, gx, fx̄, gx̄, Δ, B = init_rest(xinit,learning_function,Δ₀,ds)

    ####################
    # Run the algorithm
    ####################

    v = iterate(params) do verbose :: Function
        
        #println("x=$(norm(x,1)/length(x)), Δ=$Δ, gx=$gx")

        #p = dogleg_box(x,gx,B,Δ) # solve tr subproblem
        p = dogbox(x,gx,B,Δ)

        x̄ = x + p  # test new point

        ū,fx̄,gx̄ = learning_function(x̄,ds,Δ)
        predf = pred(B,p,gx)
        ρ = (fx-fx̄)/predf # ared/pred
        if predf == 0
            @error "Problems with step calculated"
        end
        #println("ρ=$ρ, ared = $(fx-fx̄), pred = $(predf)")

        updateBFGS!(B,gx̄-gx,p)

        if ρ < η₁               # radius update
            Δ = β₁*Δ
        elseif ρ > η₂
            if norm₂(p) > 0.8*Δ
                Δ = β₂*Δ
            end
        end

        if predf < 0
            Δ = β₁*Δ
        end

        if ρ > 1e-3
            x = x̄
            u = ū
            fx = fx̄
            gx = gx̄
        end

        ################################
        # Give function value if needed
        ################################
        v = verbose() do    
            x, u[:,:,1], fx, norm₂(gx), Δ # just show the first image on the dataset
            #fx,u[:,:,1]
        end

        v
    end

    return x, u, v
end