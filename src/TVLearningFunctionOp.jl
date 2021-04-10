
using VariationalImaging.OpDenoise
using AlgTools.LinOps
using AlgTools.Util: @threadsif, dot

export tv_op_learning_function

Primal = Array{Float64,2}
Dual = Array{Float64,3}

###################
# Aux Operators
###################
struct TOp{X} <: LinOp{X,X}
	u::Primal
	Ku::Dual
	op::LinOp
end

function TOp(u::Primal,op::LinOp)
	return TOp{Float64}(u,op(u),op)
end

function (op::TOp{X})(x::Primal; threads::Bool=true) where X
	Kx = op.op(x)
	o,m,n = size(Kx)
	res = copy(Kx)
	@threadsif threads for i=1:n
        @inbounds for j=1:m
            nKu = sqrt((op.Ku[1,i,j])^2+(op.Ku[2,i,j]^2))
			if isapprox(nKu,0)
				res[1,i,j] = Kx[1,i,j]
				res[2,i,j] = Kx[2,i,j]
			else
				res[1,i,j] = -(1/nKu)*(Kx[1,i,j]+(1/nKu^2)*(op.Ku[1,i,j]^2*Kx[1,i,j] + op.Ku[1,i,j]*op.Ku[2,i,j]*Kx[2,i,j]))
				res[2,i,j] = -(1/nKu)*(Kx[2,i,j]+(1/nKu^2)*(op.Ku[2,i,j]^2*Kx[2,i,j] + op.Ku[1,i,j]*op.Ku[2,i,j]*Kx[1,i,j]))
			end
        end
    end
    return res
end

function Base.adjoint(op::TOp{X}) where X
    return op
end

function opnorm_estimate(op::TOp{X}) where X
    return 1
end

###################
# Learning function
###################
function tv_op_learning_function(x,data)
    op = FwdGradientOp()
    u = denoise(data[2],x,op)
    cost = 0.5*norm₂²(u-data[1])
    grad = gradient(x,op,u,data[1])
    return u,cost,grad
end

function denoise(data,x::Real,op::LinOp)
    denoise_params = (
        ρ = 0,
        α = x,
        op = op,
        # PDPS
        τ₀ = 5,
        σ₀ = 0.99/5,
        accel = true,
        save_results = false,
        maxiter = 20000,
        verbose_iter = 20001,
        save_iterations = false
    )
    st_opt, iterate_opt = initialise_visualisation(false)
    opt_img = op_denoise_pdps(data; iterate=iterate_opt, params=denoise_params)
    finalise_visualisation(st_opt)
    return opt_img
end

function denoise(data,x::AbstractArray,op::LinOp)
	p = PatchOp(x,data[:,:,1]) # Adjust parameter size
	x̄ = zeros(p.size_out)
	inplace!(x̄,p,x)
    denoise_params = (
        ρ = 0,
        α = x̄,
		op = op,
        # PDPS
        τ₀ = 5,
        σ₀ = 0.99/5,
        accel = true,
        save_results = false,
        maxiter = 20000,
        verbose_iter = 20001,
        save_iterations = false
    )
    st_opt, iterate_opt = initialise_visualisation(false)
    
    opt_img = op_denoise_pdps(data; iterate=iterate_opt, params=denoise_params)
    finalise_visualisation(st_opt)
    return opt_img
end

function gradient(α::Real,op::LinOp,u::AbstractArray{T,3},ū::AbstractArray{T,3}) where T
	
	M,N,O = size(u)
	grad = 0
	for i = 1:O
		u1 = @view u[:,:,i]
		u2 = @view ū[:,:,i]
		g = gradient_reg(α,op,u1,u2)
		grad += g 
	end
	return grad
end

function gradient(α::Real, op::LinOp, u::AbstractArray{T,2}, ū::AbstractArray{T,2}) where T
	u = Float64.(Gray{Float64}.(u))
	ū = Float64.(Gray{Float64}.(ū))
	# Obtain Active and inactive sets
	n = size(u,1)
	
	# Generate centered gradient matrix
	#G = createDivMatrix(n)
	G = matrix(op,n)
	Gu = G*u[:]
	nGu = xi(Gu)
	act = nGu .< 1e-12
	inact = 1 .- act
	Act = spdiagm(0=>act)
	Inact = spdiagm(0=>inact)
	
	# Vector with grad norm in inactive components and one in the active
	den = Inact*nGu+act
	Den = spdiagm(0=>1 ./den)
	
	# prod KuKuᵗ/norm³
	prodKuKu = prodesc(Gu ./den.^3,Gu)
	
	# Adj = [spdiagm(0=>ones(n^2)) α*G';
	# 		Act*G+Inact*(prodKuKu-Den)*G Inact+eps()*Act]
	Adj = [spdiagm(0=>ones(n^2)) -G';
			Act*G+Inact*α*(prodKuKu-Den)*G Inact+eps()*Act]
	
	Track=[(-u[:]+ū[:]);zeros(2*n^2)]
	mult = Adj\Track
	p = @view mult[1:n^2]
	Gp = G*p
	gr = 0
	ν = (G'*Inact*Den*Gu)
	println("Stopping criteria: $(α*ν)")
	return p'*ν
end

function gradient_reg(α::Real,op::LinOp,u::AbstractArray{T,2}, ū::AbstractArray{T,2}) where T
	u = Float64.(Gray{Float64}.(u))
	ū = Float64.(Gray{Float64}.(ū))
	# Obtain Active and inactive sets
	n = size(u,1)
	γ = 1e8
	G = matrix(op,n)
	Gu = G*u[:]
	nGu = xi(Gu)
	act1 = nGu .- 1/γ
	act = max.(0,act1) .!= 0
	inact = 1 .- act
	Act = spdiagm(0=>act)
	Inact = spdiagm(0=>inact)
	den = Act*nGu + inact
	Den = spdiagm(0=>1 ./den)
	prodGuGu = prodesc(Gu./(den.^3),Gu)
	I = spdiagm(0=>ones(n^2))
	B = γ*Inact
	C = (Act*(prodGuGu-Den))
	p = (I+α*G'*(B-C)*G)\(ū[:]-u[:])

	return p'*(G'*(Act*Den*Gu+γ*Inact*Gu))

end

function gradient(α::AbstractArray, op::LinOp, u::AbstractArray{T,3}, ū::AbstractArray{T,3}) where T
	
	M,N,O = size(u)
	m,n = size(α)
	p = PatchOp(α,u[:,:,1]) # Adjust parameter size
	grad = zeros(size(α))
	for i = 1:O
		u1 = @view u[:,:,i]
		u2 = @view ū[:,:,i]
		g = gradient_reg(p(α),op,p,u1,u2)
		grad .+= g
	end
	return grad
end

function gradient_reg(α::AbstractArray,op::LinOp,pOp::PatchOp,u::AbstractArray{T,2}, ū::AbstractArray{T,2}) where T
	u = Float64.(Gray{Float64}.(u))
	ū = Float64.(Gray{Float64}.(ū))
	# Obtain Active and inactive sets
	m,n = size(u)
	γ = 1e8
	G = matrix(op,m)
	Gu = G*u[:]
	nGu = xi(Gu)
	act1 = nGu .- 1/γ
	act = max.(0,act1) .!= 0
	inact = 1 .- act
	Act = spdiagm(0=>act)
	Inact = spdiagm(0=>inact)
	den = Act*nGu + inact
	Den = spdiagm(0=>1 ./den)
	prodGuGu = prodesc(Gu./(den.^3),Gu)
	I = spdiagm(0=>ones(m*n))
	B = γ*Inact
	C = (Act*(prodGuGu-Den))
	p = (I+ α[:] .*G'*(B-C)*G)\(ū[:]-u[:])
	grad = reshape(spdiagm(0=>p)*(G'*(Act*Den*Gu+γ*Inact*Gu)),m,n)
	return calc_adjoint(pOp,grad)
end



function gradient(α::AbstractArray, op::LinOp, u::AbstractArray{T,2}, ū::AbstractArray{T,2}) where T
	u = Float64.(Gray{Float64}.(u))
	ū = Float64.(Gray{Float64}.(ū))
	# Obtain Active and inactive sets
	n = size(u,1)
	M,N = size(α)
	
	# Generate centered gradient matrix
	#G = createDivMatrix(n)
	G = matrix(op,n)
	Gu = G*u[:]
	nGu = xi(Gu)
	act = nGu .< 1e-12
	inact = 1 .- act
	Act = spdiagm(0=>act)
	Inact = spdiagm(0=>inact)
	
	# Vector with grad norm in inactive components and one in the active
	den = Inact*nGu+act
	Den = spdiagm(0=>1 ./den)
	
	# prod KuKuᵗ/norm³
	prodKuKu = prodesc(Gu ./den.^3,Gu)
	# Adj = [spdiagm(0=>ones(n^2)) spdiagm(0=>α[:])*G';
	# 		Act*G+Inact*(prodKuKu-Den)*G Inact+eps()*Act]
	Adj = [spdiagm(0=>ones(n^2)) G';
			Act*G+Inact*spdiagm(0=>[α[:];α[:]])*(prodKuKu-Den)*G Inact+eps()*Act]
	
	Track=[(u[:]-ū[:]);zeros(2*n^2)]
	mult = Adj\Track
	p = @view mult[1:n^2]
	return -spdiagm(0=>p[:])*(G'*Inact*Den*Gu)
end