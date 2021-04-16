
using VariationalImaging.OpDenoise
using AlgTools.LinOps
using AlgTools.Util: @threadsif, dot

export tv_op_learning_function, denoise

Primal = Array{Float64,2}
Dual = Array{Float64,3}

###################
# Learning function
###################
function tv_op_learning_function(x,data,Δ;Δt=1e-6,kwargs...)
	ū = data[1]
	f = data[2]
    op = FwdGradientOp()
    u = denoise(f,x,op;kwargs...)
	#println(maximum(u-ū))
	cost = 0.5*norm₂²(u-ū)
	if Δ > Δt
    	grad = gradient(x,op,u,ū)
	else
		grad = gradient_reg(x,op,u,ū)
	end
    return u,cost,grad
end

###################
# Denoising
###################

const denoising_default_params = (
	ρ = 0,
	# PDPS
	τ₀ = 5,
	σ₀ = 0.99/5,
	accel = true,
	save_results = false,
	maxiter = 5000,
	verbose_iter = 5001,
	save_iterations = false
)

function denoise(data,x::Real,op::LinOp;kwargs...)
    denoise_scalar_params = (
        α = x,
		op = op
    )
	params = denoising_default_params ⬿ denoise_scalar_params ⬿ kwargs
    st_opt, iterate_opt = initialise_visualisation(false)
    opt_img = op_denoise_pdps(data; iterate=iterate_opt, params=params)
    finalise_visualisation(st_opt)
    return opt_img
end

function denoise(data,x::AbstractArray,op::LinOp)
	p = PatchOp(x,data[:,:,1]) # Adjust parameter size
	x̄ = zeros(p.size_out)
	inplace!(x̄,p,x)
    denoise_patch_params = (
        α = x̄,
		op = op
    )
	params = denoising_default_params ⬿ denoise_patch_params
    st_opt, iterate_opt = initialise_visualisation(false)
    opt_img = op_denoise_pdps(data; iterate=iterate_opt, params=params)
    finalise_visualisation(st_opt)
    return opt_img
end

function gradient(α::Real,op::LinOp,u::AbstractArray{T,3},ū::AbstractArray{T,3}) where T
	
	M,N,O = size(u)
	grad = 0
	for i = 1:O
		u1 = @view u[:,:,i]
		u2 = @view ū[:,:,i]
		g = gradient(α,op,u1,u2)
		grad += g 
	end
	return grad
end

function gradient_reg(α::Real,op::LinOp,u::AbstractArray{T,3},ū::AbstractArray{T,3}) where T
	@info "Using reg gradient"
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
	
	Z1 = spzeros(2*n^2,2*n^2)
	Z2 = spzeros(n^2,2*n^2)
	# Adj = [spdiagm(0=>ones(n^2)) α*G';
	# 		Act*G+Inact*(prodKuKu-Den)*G Inact+eps()*Act]
	# Adj = [spdiagm(0=>ones(n^2)) -G';
	# 		Act*G+Inact*α*(Den-prodKuKu)*G Inact+sqrt(eps())*Act]+Inact*Den*Gu*G
	Adj = [spdiagm(0=>ones(n^2)) -G';
			Act*G+Inact*α*(Den-prodKuKu)*G Inact+eps()*Act]
	
	Track=[(u[:]-ū[:]);zeros(2*n^2)]
	mult = Adj\Track
	p = @view mult[1:n^2]
	grad = sum(scalarprod(G*p,Inact*Den*Gu))
	return -grad
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
	grad = sum(scalarprod(G*p,Act*Den*Gu+γ*Inact*Gu))
	return grad

end

function gradient(α::AbstractArray, op::LinOp, u::AbstractArray{T,3}, ū::AbstractArray{T,3}) where T
	M,N,O = size(u)
	m,n = size(α)
	p = PatchOp(α,u[:,:,1]) # Adjust parameter size
	grad = zeros(size(α))
	for i = 1:O
		u1 = @view u[:,:,i]
		u2 = @view ū[:,:,i]
		g = gradient(p(α),op,p,u1,u2)
		grad .+= g
	end
	return grad
end

function gradient_reg(α::AbstractArray, op::LinOp, u::AbstractArray{T,3}, ū::AbstractArray{T,3}) where T
	@info "Using reg gradient"
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



function gradient(α::AbstractArray, op::LinOp, pOp::PatchOp, u::AbstractArray{T,2}, ū::AbstractArray{T,2}) where T
	u = Float64.(Gray{Float64}.(u))
	ū = Float64.(Gray{Float64}.(ū))
	# Obtain Active and inactive sets
	m,n = size(u)
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
	Adj = [spdiagm(0=>ones(n^2)) -G';
			Act*G+Inact*spdiagm(0=>[α[:];α[:]])*(Den-prodKuKu)*G Inact+sqrt(eps())*Act]
	
	Track=[(u[:]-ū[:]);zeros(2*n^2)]
	mult = Adj\Track
	p = @view mult[1:n^2]
	grad = -scalarprod(G*p,Inact*Den*Gu)
	#grad = -spdiagm(0=>p[:])*(G'*Inact*Den*Gu)
	grad = reshape(grad,m,n)
	return calc_adjoint(pOp,grad)
end