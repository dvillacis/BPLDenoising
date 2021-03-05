module BPLDenoising

export generate_scalar_tv_cost, generate_cost_plot
export generate_2d_tv_cost, generate_2d_cost_plot
export scalar_bilevel_tv_learn, patch_bilevel_tv_learn
export scalar_bilevel_sumregs_learn, patch_bilevel_sumregs_learn

using Printf
using SparseArrays
using JLD2
using FileIO
using PGFPlots
pushPGFPlotsOptions("scale=0.8")
using ColorTypes: Gray
import ColorVectorSpace
using ImageContrastAdjustment
using ImageQualityIndexes
using TestDatasets

using AlgTools.Util
using AlgTools.LinOps
using AlgTools.LinkedLists
using ImageTools.Visualise

using VariationalImaging
using VariationalImaging.GradientOps
using VariationalImaging.OpDenoise

include("BilevelVisualise.jl")
include("TRBox.jl")
#include("TVLearningFunctionOp.jl")
include("TVLearningFunctionVec.jl")
include("SumRegsLearningFunction.jl")

using BPLDenoising.BilevelVisualise


const default_save_prefix = "output"

function TVDenoise(data,parameter::Real)
    denoise_params = (
        ρ = 0,
        α = parameter,
        op = FwdGradientOp(),
        # PDPS
        τ₀ = 5,
        σ₀ = 0.99/5,
        accel = true,
        save_results = false,
        maxiter = 2000,
        verbose_iter = 2001,
        save_iterations = false
    )
    st, iterate = initialise_visualisation(false)
    out = op_denoise_pdps(data; iterate=iterate, params=denoise_params)
    finalise_visualisation(st)
    return out
end

function TVDenoise(data,parameter::AbstractArray)
    p = PatchOp(parameter,data[:,:,1]) # Adjust parameter size
	x̄ = zeros(p.size_out)
	inplace!(x̄,p,parameter)
    denoise_params = (
        ρ = 0,
        α = x̄,
        op = FwdGradientOp(),
        # PDPS
        τ₀ = 5,
        σ₀ = 0.99/5,
        accel = true,
        save_results = false,
        maxiter = 2000,
        verbose_iter = 2001,
        save_iterations = false
    )
    st, iterate = initialise_visualisation(false)
    out = op_denoise_pdps(data; iterate=iterate, params=denoise_params)
    finalise_visualisation(st)
    return out
end

function L2CostFunction(u,true_)
    return 0.5*norm₂²(u-true_)
end

################################
# Scalar Cost Function Plotting
################################

function generate_cost(dataset_name, parameter_range, cost_function, denoise_function;freq=10,num_samples=1)
    true_,data = testdataset(dataset_name)
    true_ = Float64.(Gray{Float64}.(true_))[:,:,1:num_samples]
    data = Float64.(Gray{Float64}.(data))[:,:,1:num_samples]
    costs = zeros(size(parameter_range))
    iter = 1
    for i = 1:length(parameter_range)
        u = denoise_function(data,parameter_range[i])
        costs[i] = cost_function(u,true_)
        if iter%freq == 0
            @info "Denoising parameter $(parameter_range[i]): cost = $(costs[i])"
        end
        iter += 1
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
    p = Axis(Plots.Linear(parameter_range,costs,mark="none"),style="grid=both", xlabel=L"$\alpha$", ylabel=L"$\|u-\bar{u}\|^2$", title="Scalar Cost")
    PGFPlots.save(joinpath(cost_path,dataset_name*"_cost_plot.tex"),p,include_preamble=false)
    PGFPlots.save(joinpath(cost_path,dataset_name*"_cost_plot.pdf"),p)
end


function generate_scalar_tv_cost(dataset_name, parameter_range; num_samples=1)
    return generate_cost(dataset_name,parameter_range,L2CostFunction,TVDenoise; num_samples)
end

################################
# 2D Cost Function Plotting
################################

function generate_2d_cost(dataset_name, parameter_range_1, parameter_range_2, cost_function, denoise_function;freq=10,num_samples=1)
    true_,data = testdataset(dataset_name)
    true_ = Float64.(Gray{Float64}.(true_))[:,:,1:num_samples]
    data = Float64.(Gray{Float64}.(data))[:,:,1:num_samples]
    costs = zeros(length(parameter_range_1),length(parameter_range_2))
    iter = 1
    for i = 1:length(parameter_range_1)
        for j = 1:length(parameter_range_2)
            α = [parameter_range_1[i];parameter_range_2[j]] .* ones(2,1)
            u = denoise_function(data,α)
            costs[i,j] = cost_function(u,true_)
            if iter%freq == 0
                @info "Denoising parameter $α: cost = $(costs[i,j])"
            end
            iter += 1
        end
    end
    output_dir = joinpath(default_save_prefix, dataset_name)
    if isdir(output_dir) == false
        mkpath(output_dir)
    end
    @save joinpath(output_dir, dataset_name*"_cost_2d.jld2") parameter_range_1 parameter_range_2 costs
end

function generate_2d_cost_plot(dataset_name)
    cost_path = joinpath(default_save_prefix,dataset_name)

    if isdir(cost_path) == false
        @error "No cost calculation found at $cost_path"
        return false
    end

    @load joinpath(cost_path,dataset_name*"_cost_2d.jld2") parameter_range_1 parameter_range_2 costs
    c = reshape(costs,length(parameter_range_1),length(parameter_range_2))
    p = Axis(Plots.Contour(c,parameter_range_1,parameter_range_2,style="dashed",levels=46.7:0.65:52),style="grid=both", xlabel=L"$\alpha_1$", ylabel=L"$\alpha_2$", title="2D Cost")
    #p = Axis(Plots.Image(costs,(0.005,0.03),(0.005,0.03)),style="grid=both", xlabel=L"$\alpha_1$", ylabel=L"$\alpha_2$", title="2D Cost")
    PGFPlots.save(joinpath(cost_path,dataset_name*"_cost_plot_2d.tex"),p,include_preamble=false)
    PGFPlots.save(joinpath(cost_path,dataset_name*"_cost_plot_2d.pdf"),p)
end

function generate_2d_tv_cost(dataset_name, parameter_range_1, parameter_range_2;num_samples=1)
    return generate_2d_cost(dataset_name,parameter_range_1,parameter_range_2,L2CostFunction,TVDenoise;num_samples)
end


################################
# Save Experiments Results
################################

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
            @info size(b)
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


function save_results(params, b, b_data, x::AbstractArray{T,2}, opt_img, st) where T
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

function save_results(params, b, b_data, x::AbstractArray{T,3}, opt_img, st) where T
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
        x̄ = p(x)
        fn_par = (t, ext, i) -> "$(joinpath(out_path,params.save_prefix))_$(t)_$(i).$(ext)"
        adjust_histogram!(x̄,LinearStretching())
        FileIO.save(File(format"PNG", fn_par("par", "png", 1)), grayimg(x̄[:,:,1]))
        FileIO.save(File(format"PNG", fn_par("par", "png", 2)), grayimg(x̄[:,:,2]))
        FileIO.save(File(format"PNG", fn_par("par", "png", 3)), grayimg(x̄[:,:,3]))
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
    save_iterations = false,
    tol = 1e-5,
    num_samples = 1
)

const bilevel_params = (
    η₁ = 0.25,
    η₂ = 0.75,
    β₁ = 0.25,
    β₂ = 1.9,
    Δ₀ = 0.1,
    α₀ = 0.1
)

function scalar_bilevel_tv_learn(;visualise=true, save_prefix=default_save_prefix, kwargs...)
    # Parameters for this experiment
    params = default_params ⬿ bilevel_params ⬿ kwargs
    params = params ⬿ (save_prefix = "tv_optimal_parameter_scalar_" * params.dataset_name,)
    # Load dataset
    b,b_noisy = TestDatasets.testdataset(params.dataset_name)
    b = Float64.(Gray{Float64}.(b))[:,:,1:params.num_samples]
    b_noisy = Float64.(Gray{Float64}.(b_noisy))[:,:,1:params.num_samples]
    # Launch (background) visualiser
    st, iterate = initialise_bilevel_visualisation(visualise)
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
    β₂ = 1.9,
    Δ₀ = 0.0001,
    α₀ = 0.0001*ones(2,2)
)

function patch_bilevel_tv_learn(;visualise=true, save_prefix=default_save_prefix, kwargs...)
    # Parameters for this experiment
    params = default_params ⬿ patch_bilevel_params ⬿ kwargs
    params = params ⬿ (save_prefix = "tv_optimal_parameter_$(size(params.α₀))_" * params.dataset_name,)
    # Load dataset
    b,b_noisy = TestDatasets.testdataset(params.dataset_name)
    b = Float64.(Gray{Float64}.(b))[:,:,1:params.num_samples]
    b_noisy = Float64.(Gray{Float64}.(b_noisy))[:,:,1:params.num_samples]
    # Launch (background) visualiser
    st, iterate = initialise_bilevel_visualisation(visualise)
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
    β₂ = 1.9,
    Δ₀ = 0.01,
    α₀ = [0.001;0.001;0.001]
)

function scalar_bilevel_sumregs_learn(;visualise=true, save_prefix=default_save_prefix, kwargs...)
    # Parameters for this experiment
    params = default_params ⬿ sumregs_bilevel_params ⬿ kwargs
    params = params ⬿ (save_prefix = "sumregs_optimal_parameter_scalar_" * params.dataset_name,)
    # Load dataset
    b,b_noisy = TestDatasets.testdataset(params.dataset_name)
    b = Float64.(Gray{Float64}.(b))[:,:,1:params.num_samples]
    b_noisy = Float64.(Gray{Float64}.(b_noisy))[:,:,1:params.num_samples]
    # Launch (background) visualiser
    st, iterate = initialise_bilevel_visualisation(visualise)
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
    Δ₀ = 0.1,
    α₀ = 0.001*ones(2,2,3)
)

function patch_bilevel_sumregs_learn(;visualise=true, save_prefix=default_save_prefix, kwargs...)
    # Parameters for this experiment
    params = default_params ⬿ patch_sumregs_bilevel_params ⬿ kwargs
    params = params ⬿ (save_prefix = "sumregs_optimal_parameter_patch_$(size(params.α₀))" * params.dataset_name,)
    # Load dataset
    b,b_noisy = TestDatasets.testdataset(params.dataset_name)
    b = Float64.(Gray{Float64}.(b))[:,:,1:params.num_samples]
    b_noisy = Float64.(Gray{Float64}.(b_noisy))[:,:,1:params.num_samples]
    # Launch (background) visualiser
    st, iterate = initialise_bilevel_visualisation(visualise)
    # Run algorithm
    x, u, st = bilevel_learn((b,b_noisy),sumregs_learning_function; xinit=params.α₀,iterate=iterate, params=params)
    adjust_histogram!(u,LinearStretching())
    # Save results
    save_results(params, b, b_noisy, x, u, st)
    # Exit background visualiser
    finalise_visualisation(st)
end

end # Module
