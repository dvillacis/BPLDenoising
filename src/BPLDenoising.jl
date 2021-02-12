module BPLDenoising

export generate_scalar_tv_cost, generate_cost_plot
export scalar_bilevel_tv_learn, patch_bilevel_tv_learn
export scalar_bilevel_sumregs_learn, patch_bilevel_sumregs_learn

using Printf
using SparseArrays
using JLD2
using FileIO
using PGFPlots
using ColorTypes: Gray
import ColorVectorSpace
using ImageContrastAdjustment
using ImageQualityIndexes

using AlgTools.Util
using AlgTools.LinOps
using AlgTools.LinkedLists
using ImageTools.Visualise

using VariationalImaging
using VariationalImaging.TestDatasets
using VariationalImaging.GradientOps
using VariationalImaging.OpDenoise

include("Bilevel.jl")

include("TVLearningFunctionOp.jl")
include("SumRegsLearningFunction.jl")


const default_save_prefix = "output"

function TVDenoise(data,parameter)
    denoise_params = (
        ρ = 0,
        α = parameter,
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
    st, iterate = initialise_visualisation(false)
    out = op_denoise_pdps(data,op; iterate=iterate, params=denoise_params)
    finalise_visualisation(st)
    return out
end

function L2CostFunction(u,true_)
    return 0.5*norm₂²(u-true_)
end

################################
# Scalar Cost Function Plotting
################################

function generate_cost(dataset_name, parameter_range, cost_function, denoise_function)
    true_,data = testdataset(dataset_name)
    costs = zeros(size(parameter_range))
    for i = 1:length(parameter_range)
        u = denoise_function(data,parameter_range[i])
        costs[i] = cost_function(u,true_)
        @info "Denoising parameter $(parameter_range[i]): cost = $(costs[i])"
    end
    output_dir = joinpath(default_save_prefix, dataset_name)
    if isdir(output_dir) == false
        mkpath(output_dir)
    end
    @save joinpath(output_dir, dataset_name*"_cost.jld2") parameter_range costs
end

function generate_cost_plot(dataset_name)
    cost_path = joinpath(default_save_prefix,dataset_name)

    if isdir(cost_path) == false
        @error "No cost calculation found at $cost_path"
        return false
    end

    @load joinpath(cost_path,dataset_name*"_cost.jld2") parameter_range costs
    p = Axis(Plots.Linear(parameter_range,costs,mark="none"), xlabel=L"$\alpha$", ylabel=L"$\|u-\bar{u}\|^2$", title="Scalar Cost")
    PGFPlots.save(joinpath(cost_path,dataset_name*"_cost_plot.tex"),p)
    PGFPlots.save(joinpath(cost_path,dataset_name*"_cost_plot.pdf"),p)
end


function generate_scalar_tv_cost(dataset_name, parameter_range)
    return generate_cost(dataset_name,parameter_range,L2CostFunction,TVDenoise)
end


function save_results(params, b, b_data, x::Union{Real,AbstractVector{Float64}}, opt_img, st)
    if params.save_results
        out_path = joinpath(default_save_prefix,params.dataset_name)
        if isdir(out_path) == false
            mkpath(out_path)
        end
        perffile = joinpath(out_path,params.save_prefix * ".txt")
        qualityfile = joinpath(out_path,params.save_prefix * "_quality.txt")
        println("Saving " * perffile)
        write_log(perffile, st.log, "# params = $(params), x = $x\n")
        open(qualityfile,"w") do io
            write(io,"img_num \t orig_ssim \t orig_psnr \t out_ssim \t out_psnr\n")
            M,N,O = size(b)
            for i = 1:O
                noisy_ssim = assess_ssim(b[:,:,i],b_data[:,:,i])
                noisy_psnr = assess_psnr(b[:,:,i],b_data[:,:,i])
                out_ssim = assess_ssim(b[:,:,i],opt_img[:,:,i])
                out_psnr = assess_psnr(b[:,:,i],opt_img[:,:,i])
                write(io,"$i\t $noisy_ssim \t $noisy_psnr \t $out_ssim \t $out_psnr\n")

                fn = (t, ext, i) -> "$(joinpath(out_path,params.save_prefix))_$(t)_$(i).$(ext)"
                FileIO.save(File(format"PNG", fn("true", "png", i)), grayimg(b[:,:,i]))
                FileIO.save(File(format"PNG", fn("data", "png", i)), grayimg(b_data[:,:,i]))
                FileIO.save(File(format"PNG", fn("reco", "png", i)), grayimg(opt_img[:,:,i]))
            end
        end
    end
end


function save_results(params, b, b_data, x::AbstractArray, opt_img, st)
    if params.save_results
        out_path = joinpath(default_save_prefix,params.dataset_name)
        if isdir(out_path) == false
            mkpath(out_path)
        end
        perffile = joinpath(out_path,params.save_prefix * ".txt")
        qualityfile = joinpath(out_path,params.save_prefix * "_quality.txt")
        println("Saving " * perffile)
        write_log(perffile, st.log, "# params = $(params), x = $x\n")
        open(qualityfile,"w") do io
            write(io,"img_num \t orig_ssim \t orig_psnr \t out_ssim \t out_psnr\n")
            M,N,O = size(b)
            for i = 1:O
                noisy_ssim = assess_ssim(b[:,:,i],b_data[:,:,i])
                noisy_psnr = assess_psnr(b[:,:,i],b_data[:,:,i])
                out_ssim = assess_ssim(b[:,:,i],opt_img[:,:,i])
                out_psnr = assess_psnr(b[:,:,i],opt_img[:,:,i])
                write(io,"$i\t $noisy_ssim \t $noisy_psnr \t $out_ssim \t $out_psnr\n")

                fn = (t, ext, i) -> "$(joinpath(out_path,params.save_prefix))_$(t)_$(i).$(ext)"
                FileIO.save(File(format"PNG", fn("true", "png", i)), grayimg(b[:,:,i]))
                FileIO.save(File(format"PNG", fn("data", "png", i)), grayimg(b_data[:,:,i]))
                FileIO.save(File(format"PNG", fn("reco", "png", i)), grayimg(opt_img[:,:,i]))
            end
        end
        p = PatchOp(x,b[:,:,1]) # Adjust parameter size
        x̄ = zeros(p.size_out)
        inplace!(x̄,p,x)
        fn_par = (t, ext) -> "$(joinpath(out_path,params.save_prefix))_$(t).$(ext)"
        adjust_histogram!(x̄,LinearStretching())
        FileIO.save(File(format"PNG", fn_par("par", "png")), grayimg(x̄))
    end
end


###########################
# Scalar Bilevel Experiment
###########################

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
    β₂ = 1.5,
    Δ₀ = 1.0,
    α₀ = 2.0
)

function scalar_bilevel_tv_learn(;visualise=true, save_prefix=default_save_prefix, kwargs...)
    # Parameters for this experiment
    params = default_params ⬿ bilevel_params ⬿ kwargs
    params = params ⬿ (save_prefix = "tv_optimal_parameter_scalar_" * params.dataset_name,)
    # Load dataset
    b,b_noisy = TestDatasets.testdataset(params.dataset_name)
    b = Float64.(Gray{Float64}.(b))
    b_noisy = Float64.(Gray{Float64}.(b_noisy))
    # Launch (background) visualiser
    st, iterate = initialise_visualisation(visualise)
    # Run algorithm
    x, u, st = bilevel_learn((b,b_noisy),tv_op_learning_function; xinit=params.α₀,iterate=iterate, params=params)
    adjust_histogram!(u,LinearStretching())
    # Save results
    save_results(params, b, b_noisy, x, u, st)
    # Exit background visualiser
    finalise_visualisation(st)
end

###########################
# Patch Bilevel Experiment
###########################

const patch_bilevel_params = (
    η₁ = 0.25,
    η₂ = 0.75,
    β₁ = 0.25,
    β₂ = 1.5,
    Δ₀ = 1.0,
    α₀ = 0.001*ones(2,2)
)

function patch_bilevel_tv_learn(;visualise=true, save_prefix=default_save_prefix, kwargs...)
    # Parameters for this experiment
    params = default_params ⬿ patch_bilevel_params ⬿ kwargs
    params = params ⬿ (save_prefix = "tv_optimal_parameter_$(size(params.α₀))_" * params.dataset_name,)
    # Load dataset
    b,b_noisy = TestDatasets.testdataset(params.dataset_name)
    b = Float64.(Gray{Float64}.(b))
    b_noisy = Float64.(Gray{Float64}.(b_noisy))
    # Launch (background) visualiser
    st, iterate = initialise_visualisation(visualise)
    # Run algorithm
    x, u, st = bilevel_learn((b,b_noisy),tv_op_learning_function; xinit=params.α₀,iterate=iterate, params=params)
    adjust_histogram!(u,LinearStretching())
    # Save results
    save_results(params, b, b_noisy, x, u, st)
    # Exit background visualiser
    finalise_visualisation(st)
end


###########################
# Scalar Sumregs Bilevel Experiment
###########################

const sumregs_bilevel_params = (
    η₁ = 0.25,
    η₂ = 0.75,
    β₁ = 0.25,
    β₂ = 1.5,
    Δ₀ = 1.0,
    α₀ = [0.01;0.01;0.01]
)

function scalar_bilevel_sumregs_learn(;visualise=true, save_prefix=default_save_prefix, kwargs...)
    # Parameters for this experiment
    params = default_params ⬿ sumregs_bilevel_params ⬿ kwargs
    params = params ⬿ (save_prefix = "sumregs_optimal_parameter_scalar_" * params.dataset_name,)
    # Load dataset
    b,b_noisy = TestDatasets.testdataset(params.dataset_name)
    b = Float64.(Gray{Float64}.(b))
    b_noisy = Float64.(Gray{Float64}.(b_noisy))
    # Launch (background) visualiser
    st, iterate = initialise_visualisation(visualise)
    # Run algorithm
    x, u, st = bilevel_learn((b,b_noisy),sumregs_learning_function; xinit=params.α₀,iterate=iterate, params=params)
    adjust_histogram!(u,LinearStretching())
    # Save results
    save_results(params, b, b_noisy, x, u, st)
    # Exit background visualiser
    finalise_visualisation(st)
end

###########################
# Patch Sumregs Bilevel Experiment
###########################

const patch_sumregs_bilevel_params = (
    η₁ = 0.25,
    η₂ = 0.75,
    β₁ = 0.25,
    β₂ = 1.5,
    Δ₀ = 1.0,
    α₀ = [0.1*ones(2,2);0.1*ones(2,2);0.1*ones(2,2)]
)

function patch_bilevel_sumregs_learn(;visualise=true, save_prefix=default_save_prefix, kwargs...)
    # Parameters for this experiment
    params = default_params ⬿ patch_sumregs_bilevel_params ⬿ kwargs
    params = params ⬿ (save_prefix = "sumregs_optimal_parameter_scalar_" * params.dataset_name,)
    # Load dataset
    b,b_noisy = TestDatasets.testdataset(params.dataset_name)
    b = Float64.(Gray{Float64}.(b))
    b_noisy = Float64.(Gray{Float64}.(b_noisy))
    # Launch (background) visualiser
    st, iterate = initialise_visualisation(visualise)
    # Run algorithm
    x, u, st = bilevel_learn((b,b_noisy),sumregs_learning_function; xinit=params.α₀,iterate=iterate, params=params)
    adjust_histogram!(u,LinearStretching())
    # Save results
    save_results(params, b, b_noisy, x, u, st)
    # Exit background visualiser
    finalise_visualisation(st)
end

end # Module
