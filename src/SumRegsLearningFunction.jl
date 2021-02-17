
using VariationalImaging.SumRegsDenoise
using AlgTools.LinOps

###################
# Learning function
###################
function sumregs_learning_function(x::AbstractVector{Float64},data)
	x₁ = x[1]
	x₂ = x[2]
	x₃ = x[3]
	op₁ = FwdGradientOp()
	op₂ = BwdGradientOp()
	op₃ = CenteredGradientOp()
    u = denoise(data[2],x₁,x₂,x₃,op₁,op₂,op₃)
    cost = 0.5*norm₂²(u-data[1])
    grad = gradient(x₁,x₂,x₃,op₁,op₂,op₃,u,data[1])
    return u,cost,grad
end

function sumregs_learning_function(x::AbstractArray,data)
	m,n = size(x)
	x₁ = x[1:n,1:n]
	x₂ = x[n+1:2*n,1:n]
	x₃ = x[2*n+1:3*n,1:n]
	op₁ = FwdGradientOp()
	op₂ = BwdGradientOp()
	op₃ = CenteredGradientOp()
    u = denoise(data[2],x₁,x₂,x₃,op₁,op₂,op₃)
    cost = 0.5*norm₂²(u-data[1])
    grad = gradient(x₁,x₂,x₃,op₁,op₂,op₃,u,data[1])
    return u,cost,grad
end

function denoise(data,x₁::Real,x₂::Real,x₃::Real,op₁::LinOp,op₂::LinOp,op₃::LinOp)
    denoise_params = (
        ρ = 0,
        α₁ = x₁,
        α₂ = x₂,
        α₃ = x₃,
		op₁ = op₁,
		op₂ = op₂,
		op₃ = op₃,
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
    
    opt_img = sumregs_denoise_pdps(data; iterate=iterate_opt, params=denoise_params)
    finalise_visualisation(st_opt)
    return opt_img
end

function denoise(data,x₁::AbstractArray,x₂::AbstractArray,x₃::AbstractArray,op₁::LinOp,op₂::LinOp,op₃::LinOp)
    denoise_params = (
        ρ = 0,
        α₁ = x₁,
        α₂ = x₂,
        α₃ = x₃,
		op₁ = op₁,
		op₂ = op₂,
		op₃ = op₃,
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
    
    opt_img = sumregs_denoise_pdps(data; iterate=iterate_opt, params=denoise_params)
    finalise_visualisation(st_opt)
    return opt_img
end

function gradient(x₁::Real,x₂::Real,x₃::Real,op₁::LinOp,op₂::LinOp,op₃::LinOp,u::AbstractArray{T,3},ū::AbstractArray{T,3}) where T
	
	M,N,O = size(u)
	grad = zeros(3)
	for i = 1:O
		u1 = @view u[:,:,i]
		u2 = @view ū[:,:,i]
		g = gradient_reg(x₁,x₂,x₃,op₁,op₂,op₃,u1,u2)
		grad += g
	end
	return grad
end

function gradient(x₁::AbstractArray,x₂::AbstractArray,x₃::AbstractArray,op₁::LinOp,op₂::LinOp,op₃::LinOp,u::AbstractArray{T,3},ū::AbstractArray{T,3}) where T
	
	M,N,O = size(u)
	m,n = size(x₁)
	p = PatchOp(x₁,u[:,:,1]) # Adjust parameter size
	x̄₁ = zeros(p.size_out)
	x̄₂ = zeros(p.size_out)
	x̄₃ = zeros(p.size_out)
	inplace!(x̄₁,p,x₁)
	inplace!(x̄₂,p,x₂)
	inplace!(x̄₃,p,x₃)
	grad = zeros(3*M*N)
	for i = 1:O
		u1 = @view u[:,:,i]
		u2 = @view ū[:,:,i]
		g = gradient_reg(x̄₁,x̄₂,x̄₃,op₁,op₂,op₃,u1,u2)
		grad += g
	end
	grad = reshape(grad,3*M,N)
	grad₁ = grad[1:M,1:N]
	grad₂ = grad[M+1:2*M,1:N]
	grad₃ = grad[2*M+1:3*M,1:N]
	grad₁₊ = zeros(size(x₁))
	grad₂₊ = zeros(size(x₂))
	grad₃₊ = zeros(size(x₃))
	inplace!(grad₁₊,p',grad₁)
	inplace!(grad₂₊,p',grad₂)
	inplace!(grad₃₊,p',grad₃)
	return [grad₁₊;grad₂₊;grad₃₊]
end

function gradient(x₁::Real,x₂::Real,x₃::Real,op₁::LinOp,op₂::LinOp,op₃::LinOp,u::AbstractArray{T,2},ū::AbstractArray{T,2}) where T
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
	
	Adj = [spdiagm(0=>ones(n^2)) x₁*G₁' x₂*G₃' x₃*G₃';
			Act₁*G₁+Inact₁*(prodKuKu₁-Den₁)*G₁ Inact₁+eps()*Act₁ spzeros(2*n^2,2*n^2) spzeros(2*n^2,2*n^2);
			Act₂*G₂+Inact₂*(prodKuKu₂-Den₂)*G₂ spzeros(2*n^2,2*n^2) Inact₂+eps()*Act₂ spzeros(2*n^2,2*n^2);
			Act₃*G₃+Inact₃*(prodKuKu₃-Den₃)*G₃ spzeros(2*n^2,2*n^2) spzeros(2*n^2,2*n^2) Inact₃+eps()*Act₃]
	
	Track=[(u[:]-ū[:]);zeros(6*n^2)]
	mult = Adj\Track
	p = @view mult[1:n^2]
	return -[p'*(G₁'*Inact₁*Den₁*Gu₁);p'*(G₂'*Inact₂*Den₂*Gu₂);p'*(G₃'*Inact₃*Den₃*Gu₃)] 
end

function gradient_reg(x₁::Real,x₂::Real,x₃::Real,op₁::LinOp,op₂::LinOp,op₃::LinOp,u::AbstractArray{T,2},ū::AbstractArray{T,2}) where T
	u = Float64.(Gray{Float64}.(u))
	ū = Float64.(Gray{Float64}.(ū))
	# Obtain Active and inactive sets
	n = size(u,1)
	γ = 1e8
	
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


	p = (I+x₁*G₁'*(B₁-C₁)*G₁+x₂*G₂'*(B₂-C₂)*G₂+x₃*G₃'*(B₃-C₃)*G₃)\(ū[:]-u[:])

	return [p'*(G₁'*(Act₁*Den₁*Gu₁+γ*Inact₁*Gu₁));p'*(G₂'*(Act₂*Den₂*Gu₂+γ*Inact₂*Gu₂));p'*(G₃'*(Act₃*Den₃*Gu₃+γ*Inact₃*Gu₃))]
end

function gradient(x₁::AbstractArray,x₂::AbstractArray,x₃::AbstractArray,op₁::LinOp,op₂::LinOp,op₃::LinOp,u::AbstractArray{T,2},ū::AbstractArray{T,2}) where T
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
	
	Adj = [spdiagm(0=>ones(n^2)) spdiagm(0=>x₁[:])*G₁' spdiagm(0=>x₂[:])*G₃' spdiagm(0=>x₃[:])*G₃';
			Act₁*G₁+Inact₁*(prodKuKu₁-Den₁)*G₁ Inact₁+eps()*Act₁ spzeros(2*n^2,2*n^2) spzeros(2*n^2,2*n^2);
			Act₂*G₂+Inact₂*(prodKuKu₂-Den₂)*G₂ spzeros(2*n^2,2*n^2) Inact₂+eps()*Act₂ spzeros(2*n^2,2*n^2);
			Act₃*G₃+Inact₃*(prodKuKu₃-Den₃)*G₃ spzeros(2*n^2,2*n^2) spzeros(2*n^2,2*n^2) Inact₃+eps()*Act₃]
	
	Track=[(u[:]-ū[:]);zeros(6*n^2)]
	mult = Adj\Track
	p = @view mult[1:n^2]
	return [-spdiagm(0=>p[:])*(G₁'*Inact₁*Den₁*Gu₁);-spdiagm(0=>p[:])*(G₂'*Inact₂*Den₂*Gu₂);-spdiagm(0=>p[:])*(G₃'*Inact₃*Den₃*Gu₃)] 
end

function gradient_reg(x₁::AbstractArray,x₂::AbstractArray,x₃::AbstractArray,op₁::LinOp,op₂::LinOp,op₃::LinOp,u::AbstractArray{T,2},ū::AbstractArray{T,2}) where T
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
	B₃ = γ*Inact₃
	C₃ = (Act₃*(prodGuGu₃-Den₃))


	p = (I+spdiagm(0=>x₁[:])*G₁'*(B₁-C₁)*G₁+spdiagm(0=>x₂[:])*G₂'*(B₂-C₂)*G₂+spdiagm(0=>x₃[:])*G₃'*(B₃-C₃)*G₃)\(ū[:]-u[:])

	return [spdiagm(0=>p[:])*(G₁'*(Act₁*Den₁*Gu₁+γ*Inact₁*Gu₁));spdiagm(0=>p[:])*(G₂'*(Act₂*Den₂*Gu₂+γ*Inact₂*Gu₂));spdiagm(0=>p[:])*(G₃'*(Act₃*Den₃*Gu₃+γ*Inact₃*Gu₃))]
end