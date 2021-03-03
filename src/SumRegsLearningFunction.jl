
using VariationalImaging.SumRegsDenoise
using AlgTools.LinOps

###################
# Learning function
###################
function sumregs_learning_function(x::AbstractVector{Float64},data,Δ;Δt=1e-3)
	op₁ = FwdGradientOp()
	op₂ = BwdGradientOp()
	op₃ = CenteredGradientOp()
    u = sumregs_denoise(data[2],x,op₁,op₂,op₃)
    cost = 0.5*norm₂²(u-data[1])
	if Δ > Δt
    	grad = sumregs_gradient(x,op₁,op₂,op₃,u,data[1])
	else
		grad = sumregs_gradient_reg(x,op₁,op₂,op₃,u,data[1])
	end
    return u,cost,grad
end

function sumregs_learning_function(x::AbstractArray{T,3},data,Δ;Δt=1e-3) where T
	op₁ = FwdGradientOp()
	op₂ = BwdGradientOp()
	op₃ = CenteredGradientOp()
	ū = data[1]
	pOp = PatchOp(x[:,:,1],ū[:,:,1])
    u = sumregs_denoise(data[2],x,op₁,op₂,op₃,pOp)
    cost = 0.5*norm₂²(u-ū)
	if Δ > Δt
    	grad = sumregs_gradient(x,op₁,op₂,op₃,pOp,u,ū)
	else
		grad = sumregs_gradient_reg(x,op₁,op₂,op₃,pOp,u,ū)
	end
    return u,cost,grad
end

function sumregs_denoise(data,x::AbstractVector{Float64},op₁::LinOp,op₂::LinOp,op₃::LinOp)
	denoise_params = (
        ρ = 0,
        α₁ = x[1],
        α₂ = x[2],
        α₃ = x[3],
		op₁ = op₁,
		op₂ = op₂,
		op₃ = op₃,
        # PDPS
        τ₀ = 5,
        σ₀ = 0.99/5,
        accel = true,
        save_results = false,
        maxiter = 2000,
        verbose_iter = 2001,
        save_iterations = false
    )
    st_opt, iterate_opt = initialise_visualisation(false)
    opt_img = sumregs_denoise_pdps(data; iterate=iterate_opt, params=denoise_params)
    finalise_visualisation(st_opt)
    return opt_img
end

function sumregs_denoise(data,x::AbstractArray{T,3},op₁::LinOp,op₂::LinOp,op₃::LinOp,pOp::PatchOp) where T
    x̄ = pOp(x)
	denoise_params = (
        ρ = 0,
        α₁ = x̄[:,:,1],
        α₂ = x̄[:,:,2],
        α₃ = x̄[:,:,3],
		op₁ = op₁,
		op₂ = op₂,
		op₃ = op₃,
        # PDPS
        τ₀ = 5,
        σ₀ = 0.99/5,
        accel = true,
        save_results = false,
        maxiter = 2000,
        verbose_iter = 2001,
        save_iterations = false
    )
    st_opt, iterate_opt = initialise_visualisation(false)
    opt_img = sumregs_denoise_pdps(data; iterate=iterate_opt, params=denoise_params)
    finalise_visualisation(st_opt)
    return opt_img
end

function sumregs_gradient(x::AbstractVector{Float64},op₁::LinOp,op₂::LinOp,op₃::LinOp,u::AbstractArray{T,3},ū::AbstractArray{T,3}) where T
	M,N,O = size(u)
	grad = zeros(size(x))
	for i = 1:O
		u1 = @view u[:,:,i]
		u2 = @view ū[:,:,i]
		g = sumregs_gradient(x,op₁,op₂,op₃,u1,u2)
		grad += g
	end
	return grad
end

function sumregs_gradient_reg(x::AbstractVector{Float64},op₁::LinOp,op₂::LinOp,op₃::LinOp,u::AbstractArray{T,3},ū::AbstractArray{T,3}) where T
	M,N,O = size(u)
	grad = zeros(size(x))
	for i = 1:O
		u1 = @view u[:,:,i]
		u2 = @view ū[:,:,i]
		g = sumregs_gradient_reg(x,op₁,op₂,op₃,u1,u2)
		grad += g
	end
	return grad
end

function sumregs_gradient_reg(x::AbstractVector{Float64},op₁::LinOp,op₂::LinOp,op₃::LinOp,u::AbstractArray{T,2},ū::AbstractArray{T,2}) where T
	@info "Using reg gradient"
	u = Float64.(Gray{Float64}.(u))
	ū = Float64.(Gray{Float64}.(ū))
	# Obtain Active and inactive sets
	n = size(u,1)
	γ = 1e3
	
	G₁ = matrix(op₁,n)
	Gu₁ = G₁*u[:]
	nGu₁ = xi(Gu₁)
	act1₁ = nGu₁ .- 1/γ
	act₁ = max.(0,act1₁) .!= 0
	inact₁ = 1 .- act₁
	Act₁ = spdiagm(0=>act₁)
	Inact₁ = spdiagm(0=>inact₁)
	den₁ = Act₁*nGu₁ + inact₁
	Den₁ = spdiagm(0=>1 ./den₁)
	prodGuGu₁ = prodesc(Gu₁./(den₁.^3),Gu₁)
	I = spdiagm(0=>ones(n^2))
	B₁ = γ*Inact₁
	C₁ = (Act₁*(prodGuGu₁-Den₁))

	G₂ = matrix(op₂,n)
	Gu₂ = G₂*u[:]
	nGu₂ = xi(Gu₂)
	act1₂ = nGu₂ .- 1/γ
	act₂ = max.(0,act1₂) .!= 0
	inact₂ = 1 .- act₂
	Act₂ = spdiagm(0=>act₂)
	Inact₂ = spdiagm(0=>inact₂)
	den₂ = Act₂*nGu₂ + inact₂
	Den₂ = spdiagm(0=>1 ./den₂)
	prodGuGu₂ = prodesc(Gu₂./(den₂.^3),Gu₂)
	I = spdiagm(0=>ones(n^2))
	B₂ = γ*Inact₂
	C₂ = (Act₂*(prodGuGu₂-Den₂))

	G₃ = matrix(op₃,n)
	Gu₃ = G₃*u[:]
	nGu₃ = xi(Gu₃)
	act1₃ = nGu₃ .- 1/γ
	act₃ = max.(0,act1₃) .!= 0
	inact₃ = 1 .- act₃
	Act₃ = spdiagm(0=>act₃)
	Inact₃ = spdiagm(0=>inact₃)
	den₃ = Act₃*nGu₃ + inact₃
	Den₃ = spdiagm(0=>1 ./den₃)
	prodGuGu₃ = prodesc(Gu₃./(den₃.^3),Gu₃)
	I = spdiagm(0=>ones(n^2))
	B₃ = γ*Inact₃
	C₃ = (Act₃*(prodGuGu₃-Den₃))

	p = (I+x[1]*G₁'*(B₁-C₁)*G₁+x[2]*G₂'*(B₂-C₂)*G₂+x[3]*G₃'*(B₃-C₃)*G₃)\(ū[:]-u[:])
	grad = [p'*(G₁'*(Act₁*Den₁*Gu₁+γ*Inact₁*Gu₁));p'*(G₂'*(Act₂*Den₂*Gu₂+γ*Inact₂*Gu₂));p'*(G₃'*(Act₃*Den₃*Gu₃+γ*Inact₃*Gu₃))]
	return grad
end

function sumregs_gradient(x::AbstractArray{T,3},op₁::LinOp,op₂::LinOp,op₃::LinOp,pOp::PatchOp,u::AbstractArray{T,3},ū::AbstractArray{T,3}) where T
	
	M,N,O = size(u)
	grad = zeros(size(x))
	for i = 1:O
		u1 = @view u[:,:,i]
		u2 = @view ū[:,:,i]
		g = sumregs_gradient(x,op₁,op₂,op₃,pOp,u1,u2)
		grad += g
	end
	return grad
end

function sumregs_gradient_reg(x::AbstractArray{T,3},op₁::LinOp,op₂::LinOp,op₃::LinOp,pOp::PatchOp,u::AbstractArray{T,3},ū::AbstractArray{T,3}) where T
	
	M,N,O = size(u)
	grad = zeros(size(x))
	for i = 1:O
		u1 = @view u[:,:,i]
		u2 = @view ū[:,:,i]
		g = sumregs_gradient_reg(x,op₁,op₂,op₃,pOp,u1,u2)
		grad += g
	end
	return grad
end

function sumregs_gradient_reg(x::AbstractArray{T,3},op₁::LinOp,op₂::LinOp,op₃::LinOp,pOp::PatchOp,u::AbstractArray{T,2},ū::AbstractArray{T,2}) where T
	u = Float64.(Gray{Float64}.(u))
	ū = Float64.(Gray{Float64}.(ū))
	# Obtain Active and inactive sets
	m,n = size(u)
	γ = 1e8
	
	I = spdiagm(0=>ones(m*n))

	G₁ = matrix(op₁,m)
	Gu₁ = G₁*u[:]
	nGu₁ = xi(Gu₁)
	act1₁ = nGu₁ .- 1/γ
	act₁ = max.(0,act1₁) .!= 0
	inact₁ = 1 .- act₁
	Act₁ = spdiagm(0=>act₁)
	Inact₁ = spdiagm(0=>inact₁)
	den₁ = Act₁*nGu₁ + inact₁
	Den₁ = spdiagm(0=>1 ./den₁)
	prodGuGu₁ = prodesc(Gu₁./(den₁.^3),Gu₁)
	B₁ = γ*Inact₁
	C₁ = (Act₁*(prodGuGu₁-Den₁))

	G₂ = matrix(op₂,m)
	Gu₂ = G₂*u[:]
	nGu₂ = xi(Gu₂)
	act1₂ = nGu₂ .- 1/γ
	act₂ = max.(0,act1₂) .!= 0
	inact₂ = 1 .- act₂
	Act₂ = spdiagm(0=>act₂)
	Inact₂ = spdiagm(0=>inact₂)
	den₂ = Act₂*nGu₂ + inact₂
	Den₂ = spdiagm(0=>1 ./den₂)
	prodGuGu₂ = prodesc(Gu₂./(den₂.^3),Gu₂)
	B₂ = γ*Inact₂
	C₂ = (Act₂*(prodGuGu₂-Den₂))

	G₃ = matrix(op₃,m)
	Gu₃ = G₃*u[:]
	nGu₃ = xi(Gu₃)
	act1₃ = nGu₃ .- 1/γ
	act₃ = max.(0,act1₃) .!= 0
	inact₃ = 1 .- act₃
	Act₃ = spdiagm(0=>act₃)
	Inact₃ = spdiagm(0=>inact₃)
	den₃ = Act₃*nGu₃ + inact₃
	Den₃ = spdiagm(0=>1 ./den₃)
	prodGuGu₃ = prodesc(Gu₃./(den₃.^3),Gu₃)
	B₃ = γ*Inact₃
	C₃ = (Act₃*(prodGuGu₃-Den₃))
	
	x₁ = pOp(x[:,:,1])
	x₂ = pOp(x[:,:,2])
	x₃ = pOp(x[:,:,3])

	p = (I + x₁[:] .*G₁'*(B₁-C₁)*G₁ + x₂[:] .*G₂'*(B₂-C₂)*G₂ + x₃[:] .*G₃'*(B₃-C₃)*G₃)\(ū[:]-u[:])
	g₁ = reshape(spdiagm(0=>p)*(G₁'*(Act₁*Den₁*Gu₁+γ*Inact₁*Gu₁)),m,n)
	g₂ = reshape(spdiagm(0=>p)*(G₂'*(Act₂*Den₂*Gu₂+γ*Inact₂*Gu₂)),m,n)
	g₃ = reshape(spdiagm(0=>p)*(G₃'*(Act₃*Den₃*Gu₃+γ*Inact₃*Gu₃)),m,n)

	gx = zeros(pOp.size_in...,3)

	gx[:,:,1] = calc_adjoint(pOp,g₁)
	gx[:,:,2] = calc_adjoint(pOp,g₂)
	gx[:,:,3] = calc_adjoint(pOp,g₃)

	return gx
end

function sumregs_gradient(x::AbstractVector{Float64},op₁::LinOp,op₂::LinOp,op₃::LinOp,u::AbstractArray{T,2},ū::AbstractArray{T,2}) where T
	u = Float64.(Gray{Float64}.(u))
	ū = Float64.(Gray{Float64}.(ū))
	# Obtain Active and inactive sets
	n = size(u,1)
	
	# Generate centered gradient matrix fwd
	G₁ = matrix(op₁,n)
	Gu₁ = G₁*u[:]
	nGu₁ = xi(Gu₁)
	act₁ = nGu₁ .< 1e-12
	inact₁ = 1 .- act₁
	Act₁ = spdiagm(0=>act₁)
	Inact₁ = spdiagm(0=>inact₁)

	## Vector with grad norm in inactive components and one in the active
	den₁ = Inact₁*nGu₁+act₁
	Den₁ = spdiagm(0=>1 ./den₁)
	
	## prod KuKuᵗ/norm³
	prodKuKu₁ = prodesc(Gu₁ ./den₁.^3,Gu₁)

	# Generate centered gradient matrix fwd
	G₂ = matrix(op₂,n)
	Gu₂ = G₂*u[:]
	nGu₂ = xi(Gu₂)
	act₂ = nGu₂ .< 1e-12
	inact₂ = 1 .- act₂
	Act₂ = spdiagm(0=>act₂)
	Inact₂ = spdiagm(0=>inact₂)

	## Vector with grad norm in inactive components and one in the active
	den₂ = Inact₂*nGu₂+act₂
	Den₂ = spdiagm(0=>1 ./den₂)
	
	## prod KuKuᵗ/norm³
	prodKuKu₂ = prodesc(Gu₂ ./den₂.^3,Gu₂)

	# Generate centered gradient matrix fwd
	G₃ = matrix(op₃,n)
	Gu₃ = G₃*u[:]
	nGu₃ = xi(Gu₃)
	act₃ = nGu₃ .< 1e-12
	inact₃ = 1 .- act₃
	Act₃ = spdiagm(0=>act₃)
	Inact₃ = spdiagm(0=>inact₃)
	
	## Vector with grad norm in inactive components and one in the active
	den₃ = Inact₃*nGu₃+act₃
	Den₃ = spdiagm(0=>1 ./den₃)
	
	## prod KuKuᵗ/norm³
	prodKuKu₃ = prodesc(Gu₃ ./den₃.^3,Gu₃)
	
	Adj = [spdiagm(0=>ones(n^2)) x[1]*G₁' x[2]*G₃' x[3]*G₃';
			Act₁*G₁+Inact₁*(prodKuKu₁-Den₁)*G₁ Inact₁+eps()*Act₁ spzeros(2*n^2,2*n^2) spzeros(2*n^2,2*n^2);
			Act₂*G₂+Inact₂*(prodKuKu₂-Den₂)*G₂ spzeros(2*n^2,2*n^2) Inact₂+eps()*Act₂ spzeros(2*n^2,2*n^2);
			Act₃*G₃+Inact₃*(prodKuKu₃-Den₃)*G₃ spzeros(2*n^2,2*n^2) spzeros(2*n^2,2*n^2) Inact₃+eps()*Act₃]
	
	Track=[(u[:]-ū[:]);zeros(6*n^2)]
	mult = Adj\Track
	p = @view mult[1:n^2]
	return -[p'*(G₁'*Inact₁*Den₁*Gu₁);p'*(G₂'*Inact₂*Den₂*Gu₂);p'*(G₃'*Inact₃*Den₃*Gu₃)] 
end


function sumregs_gradient(x::AbstractArray{T,3},op₁::LinOp,op₂::LinOp,op₃::LinOp,pOp::PatchOp,u::AbstractArray{T,2},ū::AbstractArray{T,2}) where T
	u = Float64.(Gray{Float64}.(u))
	ū = Float64.(Gray{Float64}.(ū))
	# Obtain Active and inactive sets
	m,n = size(u)
	
	# Generate centered gradient matrix fwd
	G₁ = matrix(op₁,n)
	Gu₁ = G₁*u[:]
	nGu₁ = xi(Gu₁)
	act₁ = nGu₁ .< 1e-12
	inact₁ = 1 .- act₁
	Act₁ = spdiagm(0=>act₁)
	Inact₁ = spdiagm(0=>inact₁)

	## Vector with grad norm in inactive components and one in the active
	den₁ = Inact₁*nGu₁+act₁
	Den₁ = spdiagm(0=>1 ./den₁)
	
	## prod KuKuᵗ/norm³
	prodKuKu₁ = prodesc(Gu₁ ./den₁.^3,Gu₁)

	# Generate centered gradient matrix fwd
	G₂ = matrix(op₂,n)
	Gu₂ = G₂*u[:]
	nGu₂ = xi(Gu₂)
	act₂ = nGu₂ .< 1e-12
	inact₂ = 1 .- act₂
	Act₂ = spdiagm(0=>act₂)
	Inact₂ = spdiagm(0=>inact₂)

	## Vector with grad norm in inactive components and one in the active
	den₂ = Inact₂*nGu₂+act₂
	Den₂ = spdiagm(0=>1 ./den₂)
	
	## prod KuKuᵗ/norm³
	prodKuKu₂ = prodesc(Gu₂ ./den₂.^3,Gu₂)

	# Generate centered gradient matrix fwd
	G₃ = matrix(op₃,n)
	Gu₃ = G₃*u[:]
	nGu₃ = xi(Gu₃)
	act₃ = nGu₃ .< 1e-12
	inact₃ = 1 .- act₃
	Act₃ = spdiagm(0=>act₃)
	Inact₃ = spdiagm(0=>inact₃)
	
	## Vector with grad norm in inactive components and one in the active
	den₃ = Inact₃*nGu₃+act₃
	Den₃ = spdiagm(0=>1 ./den₃)
	
	## prod KuKuᵗ/norm³
	prodKuKu₃ = prodesc(Gu₃ ./den₃.^3,Gu₃)

	x₁ = pOp(x[:,:,1])
	x₂ = pOp(x[:,:,2])
	x₃ = pOp(x[:,:,3])

	Adj = [spdiagm(0=>ones(n^2)) spdiagm(0=>x₁[:])*G₁' spdiagm(0=>x₂[:])*G₃' spdiagm(0=>x₃[:])*G₃';
			Act₁*G₁+Inact₁*(prodKuKu₁-Den₁)*G₁ Inact₁+eps()*Act₁ spzeros(2*n^2,2*n^2) spzeros(2*n^2,2*n^2);
			Act₂*G₂+Inact₂*(prodKuKu₂-Den₂)*G₂ spzeros(2*n^2,2*n^2) Inact₂+eps()*Act₂ spzeros(2*n^2,2*n^2);
			Act₃*G₃+Inact₃*(prodKuKu₃-Den₃)*G₃ spzeros(2*n^2,2*n^2) spzeros(2*n^2,2*n^2) Inact₃+eps()*Act₃]
	
	Track=[(u[:]-ū[:]);zeros(6*n^2)]
	mult = Adj\Track
	p = @view mult[1:n^2]

	g₁ = reshape(-spdiagm(0=>p[:])*(G₁'*Inact₁*Den₁*Gu₁),m,n)
	g₂ = reshape(-spdiagm(0=>p[:])*(G₂'*Inact₂*Den₂*Gu₂),m,n)
	g₃ = reshape(-spdiagm(0=>p[:])*(G₃'*Inact₃*Den₃*Gu₃),m,n)

	gx = zeros(pOp.size_in...,3)

	gx[:,:,1] = calc_adjoint(pOp,g₁)
	gx[:,:,2] = calc_adjoint(pOp,g₂)
	gx[:,:,3] = calc_adjoint(pOp,g₃)
	return gx 
end

