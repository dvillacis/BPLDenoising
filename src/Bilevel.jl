########################################################
# Bilevel Parameter Learning via Nonsmooth Trust Region 
########################################################

__precompile__()

module Bilevel

using SparseArrays

using AlgTools
using AlgTools.Util
using AlgTools.LinOps
import AlgTools.Iterate

using VariationalImaging.GradientOps

using LinearAlgebra

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

function init_rest(x,learning_function::Function,Δ,ds)
    x̄ = copy(x)
    u,fx,gx = learning_function(x,ds)
    ū = copy(u)
    fx̄ = copy(fx)
    gx̄ = copy(gx)
    #B = ZeroOp{typeof(x)}()
    B = IdOp{typeof(x)}()
    return x, x̄, u, ū, fx, gx, fx̄, gx̄, Δ, B
end


############
# Auxiliary Functions
############

function cauchy_point_box(x::Real,Δ,g,B)
    Δmax = 10.0
    γ = min(1,Δmax/norm(g))
    t = 0
    gᵗBg = AlgTools.Util.dot(g,B(g))
    if gᵗBg ≤ 0 # Negative curvature detected
        t = (Δ/10.0)*γ
    else
        t = min(norm(g)^2/gᵗBg,(Δ/10.0)*γ)
    end
    d = -t*g
    x_ = x + d
    if x_ <= 0
       x_ = eps()
    end
    return x_-x
end

function cauchy_point_box(x::AbstractArray,Δ,g,B)
    Δmax = 10.0
    γ = min(1,Δmax/norm(g))
    t = 0
    gᵗBg = AlgTools.Util.dot(g,B(g))
    if gᵗBg ≤ 0 # Negative curvature detected
        t = (Δ/10.0)*γ
    else
        t = min(norm(g)^2/gᵗBg,(Δ/10.0)*γ)
    end
    d = -t*g
    x_ = x + d
    Px = clamp!(x_,eps(),Inf)
    return Px-x
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

    ######################
    # Initialise iterates
    ######################

    x, x̄, u, ū, fx, gx, fx̄, gx̄, Δ, B = init_rest(xinit,learning_function,Δ₀,ds)

    ####################
    # Run the algorithm
    ####################

    v = iterate(params) do verbose :: Function
        
        println("x=$(norm(x,1)/length(x)), Δ=$Δ, gx=$(norm(gx,1)/length(x))")

        p = cauchy_point_box(x,Δ,gx,B) # solve tr subproblem

        x̄ = x + p  # test new point

        ū,fx̄,gx̄ = learning_function(x̄,ds)
        ρ = (-AlgTools.Util.dot(p,gx) - 0.5*AlgTools.Util.dot(p,B(p)))/(fx-fx̄) # pred/ared

        if ρ < η₁               # radius update
            Δ = β₁*Δ
        elseif ρ > η₂
            Δ = β₂*Δ
        else
            Δ = β₁*Δ
        end

        if ρ > η₁
            x = x̄
            u = ū
            fx = fx̄
            gx = gx̄
        end

        ################################
        # Give function value if needed
        ################################
        v = verbose() do     
            fx, u[:,:,1] # just show the first image on the dataset
        end

        v
    end

    return x, u, v
end

end # Module