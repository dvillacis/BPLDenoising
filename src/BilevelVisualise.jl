##################################################
# Visualising TR Bilevel and data-collecting iteration tools
##################################################

module BilevelVisualise

using Printf
using FileIO
using Setfield
using ColorTypes: Gray
using ImageContrastAdjustment
import GR

using AlgTools.Util
using AlgTools.StructTools
using AlgTools.LinkedLists
using AlgTools.Comms

using VariationalImaging.GradientOps

##############
# Our exports
##############

export BilevelLogEntry,
       bg_bilevel_visualise,
       bilevel_visualise,
       clip,
       #grayimg,
       secs_ns,
       iterate_bilevel_visualise,
       initialise_bilevel_visualisation,
       finalise_bilevel_visualisation

##################
# Data structures
##################

struct BilevelLogEntry <: IterableStruct
    iter :: Int
    time :: Float64
    function_value :: Float64
    gradient_value :: Float64
    radius_value :: Float64
end

struct BilevelState
    vis :: Union{Channel,Bool,Nothing}
    visproc :: Union{Nothing,Task}
    start_time :: Union{Real,Nothing}
    wasted_time :: Real
    log :: LinkedList{BilevelLogEntry}
end

##################
# Helper routines
##################

@inline function secs_ns()
    return convert(Float64, time_ns())*1e-9
end

clip = x -> min(max(x, 0.0), 1.0)
grayimg = im -> Gray.(clip.(im))

################
# Visualisation
################

function process_bilevel_channel(fn, rc)
    while true
        d=take!(rc)
        # Take only the latest image to visualise
        while isready(rc)
            d=take!(rc)
        end
        # We're done if we were fed nothing
        if isnothing(d)
            break
        end
        try
            fn(d)
        catch ex
            error("Exception in process_channel handler. Terminating.\n")
            rethrow(ex)
        end 
    end
end

function bg_bilevel_visualise(rc)
    process_bilevel_channel(do_visualise, rc)
end

# function do_visualise(imgs)
#     plt = im -> plot(grayimg(im), showaxis=false, grid=false, aspect_ratio=:equal, margin=2mm)
#     display(plot([plt(imgs[i]) for i =1:length(imgs)]..., reuse=true, margin=0mm))
# end

grayGR = x -> begin
    y = round(UInt32, 0xff*clip(x))
    return 0x010101*y + 0xff000000
end

function fill_viewport(vp, c)
    GR.savestate()
    GR.selntran(0)
    GR.setscale(0)
    GR.setfillintstyle(GR.INTSTYLE_SOLID)
    GR.setfillcolorind(c)
    GR.fillrect(vp...)
    GR.selntran(1)
    GR.restorestate()
end

function do_visualise(imgs; refresh=true, fullscreen=false)
    n = length(imgs)

    # Get device dimensions in metres and pixels
    scrw, scrh, pw, ph = GR.inqdspsize()
    imgaspect = n > 0 ? float(size(imgs[1], 1))/float(size(imgs[1], 2)) : 1
    # Scaling to maximum size window
    sc=0.7
    # Set up window and transformations
    GR.clearws()
    GR.setscale(0);
    GR.selntran(1)
    # - First OS window size
    if fullscreen
        w, h = float(scrw), float(scrh)
    elseif scrw/n>scrh
        w, h = float(sc*scrh*n), float(sc*scrh*imgaspect)
    else
        w, h = float(sc*scrw), float(sc*scrw/n*imgaspect)
    end
    GR.setwsviewport(0, w, 0, h)
    # NDC to device
    if w>h
        canvas=[0, 1, 0, h/w]
    else
        canvas=[0, w/h, 0, 1]
    end
    GR.setwswindow(canvas...)
    fill_viewport(canvas, 1)
    # World coordinates to NDC
    if imgaspect/n<h/w
        y0 = (canvas[3]+canvas[4])/2
        ww = (canvas[2]-canvas[1])/2
        y1 = y0-ww*(imgaspect/n)
        y2 = y0+ww*(imgaspect/n)
        GR.setviewport(canvas[1], canvas[2], y1, y2)
    else
        x0 = (canvas[1]+canvas[2])/2
        hh = (canvas[4]-canvas[3])/2
        x1 = x0-hh*(n/imgaspect)
        x2 = x0+hh*(n/imgaspect)
        GR.setviewport(x1, x2, canvas[3], canvas[4])
    end    
    GR.setwindow(0, n, 0, 1)
    # Clear background
    # Plot images
    for i=1:n
        im = imgs[i]'
        sz = size(im)
        GR.drawimage(i-1, i, 0, 1, sz[1], sz[2], grayGR.(im))
    end
    if refresh
        GR.updatews()
    end
    
end

function bilevel_visualise(channel_or_toggle, data)
    if isa(channel_or_toggle, Channel)
        put_onlylatest!(channel_or_toggle, data)
    elseif isa(channel_or_toggle, Bool) && channel_or_toggle
        do_visualise(data)
    end
end

######################################################
# Iterator that does visualisation and log collection
######################################################

function iterate_bilevel_visualise(st :: BilevelState,
                           step :: Function,
                           params :: NamedTuple) where DisplacementT
    try
        stop_flag = false
        for iter=1:params.maxiter 
            st = step() do calc_objective
                if isnothing(st.start_time)
                    # The Julia precompiler is a miserable joke, apparently not crossing module
                    # boundaries, so only start timing after the first iteration.
                    st = @set st.start_time=secs_ns()
                end

                verb = params.verbose_iter!=0 && mod(iter, params.verbose_iter) == 0
                    
                if verb || iter ≤ 20 || (iter ≤ 200 && mod(iter, 10) == 0)
                    verb_start = secs_ns()
                    tm = verb_start - st.start_time - st.wasted_time
                    par, x, value, g, Δ = calc_objective()

                    entry = BilevelLogEntry(iter, tm, value, g, Δ)

                    # (**) Collect a singly-linked list of log to avoid array resizing
                    # while iterating
                    st = @set st.log=LinkedListEntry(entry, st.log)
                    
                    if verb
                        @printf("%d/%d x=%f, f=%f, g=%f, Δ=%f\n", iter, params.maxiter, norm₂(par), value, g, Δ)
                        if par isa Union{Real,AbstractVector}
                            bilevel_visualise(st.vis, (x,))
                        elseif par isa AbstractArray{Float64,3}
                            pOp = PatchOp(par,x)
                            par_ = adjust_histogram(pOp(par),LinearStretching())
                            bilevel_visualise(st.vis, (x,par_[:,:,1],par_[:,:,2],par_[:,:,3],))
                        else
                            pOp = PatchOp(par,x)
                            par_ = Gray.(pOp(par))
                            if abs(maximum(par_)-minimum(par)) < sqrt(eps()) 
                                par_ = (par_ .- minimum(par_)) ./ (maximum(par_))
                            else
                                par_ = (par_ .- minimum(par_)) ./ (maximum(par_)-minimum(par_))
                            end
                            bilevel_visualise(st.vis, (x,par_,))
                        end
                    end

                    if params.save_iterations
                        fn = t -> "$(params.save_prefix)_$(t)_iter$(iter).png"
                        save(File(format"PNG", fn("reco")), grayimg(x))
                    end

                    st = @set st.wasted_time += (secs_ns() - verb_start)

                    # stop criteria
                    if Δ < params.tol
                        stop_flag = true
                    end
                end
                
                return st
            end
            if stop_flag
                break
            end
        end
    catch ex
        if isa(ex, InterruptException)
            # If SIGINT is received (user pressed ^C), terminate computations,
            # returning current status. Effectively, we do not call `step()` again,
            # ending the iterations, but letting the algorithm finish up.
            # Assuming (**) above occurs atomically, `st.log` should be valid, but
            # any results returned by the algorithm itself may be partial, as for
            # reasons of efficiency we do *not* store results of an iteration until
            # the next iteration is finished.
            printstyled("\rUser interrupt—finishing up.\n", bold=true, color=202)
        else
            throw(ex)
        end
    end
    
    return st
end

####################
# Launcher routines
####################

function initialise_bilevel_visualisation(visualise; iterator=iterate_bilevel_visualise)
    # Create visualisation
    if visualise
        rc = Channel(1)
        visproc = Threads.@spawn bg_bilevel_visualise(rc)
        bind(rc, visproc)
        vis = rc
    else
        vis = false
        visproc = nothing
    end

    st = BilevelState(vis, visproc, nothing, 0.0, nothing)
    iterate = curry(iterate_bilevel_visualise, st)

    return st, iterate
end

function finalise_bilevel_visualisation(st)
    if isa(st.vis, Channel)
        # Tell subprocess to finish, and wait
        put!(st.vis, nothing)
        close(st.vis)
        wait(st.visproc)
    end
end

end # Module