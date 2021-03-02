using LinearAlgebra, ForwardDiff
using LinearOperators, Krylov

function newton_step(B,gx)
    pn, ks = Krylov.cg(B,-gx)
    if ks.solved == false
        @error ks
    end
    return pn
end

function dogleg(gx,B,Δ)
    pn = newton_step(B,gx)
    if norm(pn) <= Δ
        return pn
    end
    p = -(norm(gx)^2/(gx'*(B*gx)))*gx # Cauchy Step
    if norm(p) >= Δ
        return (p/norm(p))*Δ
    end
    proj = boundary_l2(p,pn-p,Δ)
    return proj
end

# Dogleg step calculation using l_\infty ball
function dogleg_box(x,gx,B,Δ)
    lb,ub = get_bounds(x,Δ)
    #@info lb,ub
    pn = newton_step(B,gx)
    if in_bounds(lb,Δ,pn)
        return pn
    end
    p = -(norm(gx)^2/(gx'*(B*gx)))*gx # Cauchy Step
    if in_bounds(lb,Δ,p) == false
        #@info "Scaled"
        t = step_to_bound(x,p/norm(p),lb,Δ)
        return (p/norm(p))*t
    end
    #@info "Dogleg"
    t = step_to_bound(x,pn-p,lb,Δ)
    return p + t * (pn-p)
end

# Distance to l_\infty bound
function step_to_bound(x,p,lb,ub)
    dist = max.(lb ./ p, ub ./ p)
    return minimum(dist)
end

# Test if vector is within the l_\infty ball
function in_bounds(lb,ub,x)
    return all(x .>= lb) && all(x .<= ub)
end

function get_bounds(x,Δ)
    lb = max.(-Δ,eps() .-x)
    ub = Δ * ones(size(x))
    return lb,ub
end

function boundary_l2(p,q,Δ)
    a = norm(q)^2
    b = 2*p'*q
    c = norm(p)^2 - Δ^2
    α = (-b + sqrt(b^2 - a*c))/(2*a)
    return p + α * q
end

function tr(f,∇f,x₀;maxiter=1000,tol=1e-6,Δ₀=1.0,freq=100)
    x = x₀
    fx = f(x₀)
    gx = ∇f(x₀)
    Δ = Δ₀
    #B = LinearOperators.LSR1Operator(length(x))
    B = LinearOperators.LBFGSOperator(length(x))
    iter = 1
    for i = 1:maxiter
        
        p = dogleg_box(x,gx,B,Δ) # Step calculation

        # Step update
        x̄ = x + p
		fx̄ = f(x̄)
		gx̄ = ∇f(x̄)

        # Quality evaluation
        pred = -gx'*p-0.5*p'*(B*p)
        ared = fx-fx̄
        ρ = ared/pred

        # Trust region modification
        if ρ > 0.75
            Δ = min(1e10,1.5*Δ)
        elseif ρ < 0.25
            Δ *= 0.25
        end

        # SR1 update
        y = gx̄-gx
        y_Bs = y - B*p
        if abs(p'*y_Bs) >= 1e-8 * norm(p) * norm(y_Bs)
            push!(B,p,gx̄-gx)
        end

        if ρ > 0.25
            x = x̄
            fx=fx̄
            gx = gx̄
        end

        if norm(gx) <= tol
            break
        end
        if iter%freq == 0
            @info "$i: \tx=$x,\tfx=$fx,\tgx=$(norm(gx)),\tΔ=$Δ"
        end
        iter += 1
    end
    return x,fx,norm(gx),iter
end

f(x) = (1-x[1])^2 + 100*(x[2]-x[1]^2)^2
∇f(x) = ForwardDiff.gradient(f,x)
x₀ = [10.1;10.2]
@time opt,fx,gx,iter = tr(f,∇f,x₀;maxiter=1000,freq=1)
