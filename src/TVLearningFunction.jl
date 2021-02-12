
using VariationalImaging.OpDenoise

###################
# Learning function
###################
function tv_learning_function(x,data)
    u = denoise(data[2],x)
    cost = 0.5*norm₂²(u-data[1])
    grad = gradient(x,u,data[1])
    return u,cost,grad
end

function denoise(data,x::Real)
    denoise_params = (
        ρ = 0,
        α = x,
		op = FwdGradientOp(),
        # PDPS
        τ₀ = 5,
        σ₀ = 0.99/5,
        accel = true,
        save_results = false,
        maxiter = 1000,
        verbose_iter = 1001,
        save_iterations = false
    )
    st_opt, iterate_opt = initialise_visualisation(false)
    opt_img = op_denoise_pdps(data; iterate=iterate_opt, params=denoise_params)
    finalise_visualisation(st_opt)
    return opt_img
end

function denoise(data,x::AbstractArray)
	p = PatchOp(x,data[:,:,1]) # Adjust parameter size
	x̄ = zeros(p.size_out)
	inplace!(x̄,p,x)
    denoise_params = (
        ρ = 0,
        α = x̄,
		op = FwdGradientOp(),
        # PDPS
        τ₀ = 5,
        σ₀ = 0.99/5,
        accel = true,
        save_results = false,
        maxiter = 1000,
        verbose_iter = 1001,
        save_iterations = false
    )
    st_opt, iterate_opt = initialise_visualisation(false)
    
    opt_img = op_denoise_pdps(data; iterate=iterate_opt, params=denoise_params)
    finalise_visualisation(st_opt)
    return opt_img
end

function gradient(α,u::AbstractArray{T,3},ū::AbstractArray{T,3}) where T
	
	M,N,O = size(u)
	grad = 0
	for i = 1:O
		u1 = @view u[:,:,i]
		u2 = @view ū[:,:,i]
		g = gradient(α,u1,u2)
		grad += g
	end
	return grad
end

function gradient(α::Real,u::AbstractArray{T,2},ū::AbstractArray{T,2}) where T
	u = Float64.(Gray{Float64}.(u))
	ū = Float64.(Gray{Float64}.(ū))
	# Obtain Active and inactive sets
	n = size(u,1)
	
	# Generate centered gradient matrix
	G = createDivMatrix(n)
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
	
	Adj = [spdiagm(0=>ones(n^2)) G';
			Act*G+Inact*α*(prodKuKu-Den)*G Inact+eps()*Act]
	
	Track=[(u[:]-ū[:]);zeros(2*n^2)]
	mult = Adj\Track
	p = @view mult[1:n^2]
	return -p'*(G'*Inact*Den*Gu)
end

function gradient(α::AbstractArray,u::AbstractArray{T,3},ū::AbstractArray{T,3}) where T
	
	M,N,O = size(u)
	m,n = size(α)
	p = PatchOp(α,u[:,:,1]) # Adjust parameter size
	ᾱ = zeros(p.size_out)
	inplace!(ᾱ,p,α)
	grad = zeros(size(ᾱ[:]))
	for i = 1:O
		u1 = @view u[:,:,i]
		u2 = @view ū[:,:,i]
		g = gradient(ᾱ,u1,u2)
		grad .+= g
	end
	grad = reshape(grad,M,N)
	grad₊ = zeros(size(α))
	inplace!(grad₊,p',grad)
	return grad₊
end

function gradient(α::AbstractArray,u::AbstractArray{T,2},ū::AbstractArray{T,2}) where T
	u = Float64.(Gray{Float64}.(u))
	ū = Float64.(Gray{Float64}.(ū))
	# Obtain Active and inactive sets
	n = size(u,1)
	M,N = size(α)
	
	# Generate centered gradient matrix
	G = createDivMatrix(n)
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
	Adj = [spdiagm(0=>ones(n^2)) G';
			Act*G+Inact*spdiagm(0=>[α[:];α[:]])*(prodKuKu-Den)*G Inact+eps()*Act]
	
	Track=[(u[:]-ū[:]);zeros(2*n^2)]
	mult = Adj\Track
	p = @view mult[1:n^2]
	return -spdiagm(0=>p[:])*(G'*Inact*Den*Gu)
end