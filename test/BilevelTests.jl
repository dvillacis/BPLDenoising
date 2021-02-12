__precompile__()

module BilevelTests

export test_bilevel_learn

using Printf
using SparseArrays
using FileIO
using ColorTypes: Gray
import ColorVectorSpace
using ImageContrastAdjustment

using AlgTools.Util
using AlgTools.LinOps
using AlgTools.LinkedLists
using ImageTools.Visualise

using VariationalImaging

using VariationalImaging.Util
using VariationalImaging.GradientOps
using VariationalImaging.OpDenoise

using VariationalImaging.TestDatasets
using VariationalImaging.Bilevel

# Parameters
const default_save_prefix="bilevel_result_"

const default_params = (
    verbose_iter = 1,
    maxiter = 20,
    save_results = true,
    dataset_name = "cameraman128_5",
    save_iterations = false
)

const bilevel_params = (
    η₁ = 0.25,
    η₂ = 0.75,
    β₁ = 0.25,
    β₂ = 2.0,
    Δ₀ = 1.0,
    x₀ = 0.001
)

function save_results(params, b, b_data, x, opt_img,st)
    if params.save_results
        perffile = params.save_prefix * ".txt"
        println("Saving " * perffile)
        write_log(perffile, st.log, "# params = $(params), x = $x\n")
        fn = (t, ext) -> "$(params.save_prefix)_$(t).$(ext)"
        save(File(format"PNG", fn("true", "png")), grayimg(b))
        save(File(format"PNG", fn("data", "png")), grayimg(b_data))
        save(File(format"PNG", fn("reco", "png")), grayimg(opt_img))
    end
end

###############
# Learning function
###############
function learning_function(x,data)
    op = FwdGradientOp()
    u = denoise(data[2],x,op)
    cost = 0.5*norm₂²(u-data[1])
    grad = gradient(x,u,data[1],op)
    return u,cost,grad
end

function denoise(data,x,op::LinOp)
    denoise_params = (
        ρ = 0,
        α = x,
        op=op,
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

function gradient(α,u::AbstractArray{T,3},ū::AbstractArray{T,3},op::LinOp) where T
	
	M,N,O = size(u)
	grad = 0
	for i = 1:O
		u1 = @view u[:,:,i]
		u2 = @view ū[:,:,i]
		g = gradient(α,u1,u2,op)
		grad += g
	end
	return grad
end

function gradient(α,u::AbstractArray{T,2},ū::AbstractArray{T,2},op::LinOp) where T
	u = Float64.(Gray{Float64}.(u))
	ū = Float64.(Gray{Float64}.(ū))
	# Obtain Active and inactive sets
	n = size(u,1)
	
	# Generate centered gradient matrix
    G = matrix(op,n)
	#G = createDivMatrix(n)
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
	
	Adj = [spdiagm(0=>ones(n^2)) α*G';
			Act*G+Inact*(prodKuKu-Den)*G Inact+eps()*Act]
	
	Track=[(u[:]-ū[:]);zeros(2*n^2)]
	mult = Adj\Track
	p = @view mult[1:n^2]
	return -p'*(G'*Inact*Den*Gu)
end

###############
# Bilevel learn test
###############

function test_bilevel_learn(;
    visualise=true,
    save_prefix=default_save_prefix,
    kwargs...)

    # Parameters for this experiment
    params = default_params ⬿ bilevel_params ⬿ kwargs
    params = params ⬿ (save_prefix = save_prefix * "denoise_" * params.dataset_name,)

    # Load dataset
    b,b_noisy = TestDatasets.testdataset(params.dataset_name)
    b = Float64.(Gray{Float64}.(b))
    b_noisy = Float64.(Gray{Float64}.(b_noisy))

    # Launch (background) visualiser
    st, iterate = initialise_visualisation(visualise)

    # Run algorithm
    x, u, st = bilevel_learn((b,b_noisy),learning_function; xinit=params.x₀,iterate=iterate, params=params)

    adjust_histogram!(u,LinearStretching())

    save_results(params, b, b_noisy, x, u, st)

    # Exit background visualiser
    finalise_visualisation(st)
end


end