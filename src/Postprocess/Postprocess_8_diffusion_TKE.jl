#The diffusion term of TKE budget due to pressure 
export Diffusion
@kwdef struct Diffusion{T<:AbstractField,S<:AbstractString,T1<:AbstractField,T2<:AbstractField,T3<:AbstractField,T4<:AbstractField, T5<:AbstractField}
    name::S
    meanU::T
    meanp::T1
    meanpU::T2
    meanpUfluc::T3
    meanpUflucf::T4
    result::T5
    start::Union{Real,Nothing}
    stop::Union{Real,Nothing}
    update_interval::Union{Real,Nothing}
end  

function Diffusion(field; name::AbstractString="Diffusion_TKE", start::Union{Real,Nothing}=nothing, stop::Union{Real,Nothing}=nothing,update_interval::Union{Real,Nothing}=nothing)
    if field isa VectorField
        meanU = VectorField(field.mesh)
        meanp = ScalarField(field.mesh)
        meanpU = VectorField(field.mesh)
        meanpUfluc = VectorField(field.mesh)
        meanpUflucf = FaceVectorField(field.mesh)
        result = ScalarField(field.mesh)
    else
        throw(ArgumentError("Unsupported field type: $(typeof(field))"))
    end
    return  Diffusion(name=name,meanU=meanU,meanp=meanp,meanpU=meanpU,meanpUfluc=meanpUfluc,meanpUflucf=meanpUflucf,result=result,start=start,stop=stop,update_interval=update_interval)
end


function runtime_postprocessing!(D::Diffusion,iter::Integer,n_iterations::Integer,config,Str,model,time)
    if must_calculate(D,iter,n_iterations)
        current_U = model.momentum.U
        current_p = model.momentum.p
        n = div(iter - D.start,D.update_interval) + 1

        #The pressure term 1/ρ * ∇⋅⟨p'u'⟩ 
        _update_running_mean!(D.meanU,current_U,n)
        _update_running_mean!(D.meanp,current_p,n)
        _update_running_mean!(D.meanpU,current_p * current_U ,n)
        
        tmp = D.meanpU - (D.meanU * D.meanp)
        D.meanpUfluc.x.values .= tmp.x.values
        D.meanpUfluc.y.values .= tmp.y.values
        D.meanpUfluc.z.values .= tmp.z.values
        #divergence of ⟨u'p'⟩
        interpolate!(D.meanpUflucf,D.meanpUfluc,config)
        div!(D.result,D.meanpUflucf,config)
        #still need to scale by the density 

        #The 
    end
    return nothing 
end



function convert_time_to_iterations(D::Diffusion, model,dt,iterations)
    if model.time === Transient()
        if D.start === nothing
            start = 1
        else 
            D.start >= 0  || throw(ArgumentError("Start must be a value ≥ 0 (got $(D.start))"))
            start = clamp(ceil(Int, D.start / dt), 1, iterations) 
        end

        if D.stop === nothing 
            stop = iterations
        else
            D.stop ≥ 0 || throw(ArgumentError("stop must be ≥ 0 (got $(D.stop))"))
            stop = clamp(floor(Int,D.stop / dt), 1, iterations)
        end

        if D.update_interval === nothing 
            update_interval = 1
        else
            D.update_interval > 0 || throw(ArgumentError("update interval must be > 0 (got $(D.update_interval))"))
            update_interval = max(1, floor(Int,D.update_interval / dt))
        end
        stop >= start || throw(ArgumentError("After conversion with dt=$dt the averaging window is empty (start = $start, stop = $stop)"))
        return Diffusion(name=D.name,meanU=D.meanU,meanp=D.meanp,meanpU=D.meanpU,meanpUfluc=D.meanpUfluc,meanpUflucf=D.meanpUflucf,result = D.result,start=start,stop=stop,update_interval=update_interval)

    else #for Steady runs use iterations 
        if D.start === nothing
            start = 1
        else 
            D.start isa Integer || throw(ArgumentError("For steady runs, start must be specified in iterations and therefore be an integer (got $(D.start))"))
            D.start >=1     || throw(ArgumentError("Start must be ≥1 (got $(D.start))"))
            start = D.start
        end

        if D.stop === nothing 
            stop = iterations
        else
            D.stop isa Integer || throw(ArgumentError("For steady runs, stop must be specified in iterations and therefore be an integer (got $(D.stop))"))
            D.stop >=1     || throw(ArgumentError("Stop must be ≥1 (got $(D.stop))"))
            stop = D.stop
        end

        if D.update_interval === nothing 
            update_interval = 1
        else
            D.update_interval isa Integer || throw(ArgumentError("For steady runs, update_interval must be specified in iterations and therefore be an integer (got $(D.update_interval))"))
            D.update_interval >= 1 || throw(ArgumentError("update interval must be ≥1 (got $(D.update_interval))"))
            update_interval = D.update_interval
        end

        stop >= start || throw(ArgumentError("stop iteration needs to be ≥ start  (got start = $start, stop = $stop)"))
        stop <= iterations || throw(ArgumentError("stop ($stop) must be ≤ iterations ($iterations)"))
        return Diffusion(name=D.name,meanU=D.meanU,meanp=D.meanp,meanpU=D.meanpU,meanpUfluc=D.meanpUfluc,meanpUflucf=D.meanpUflucf,result = D.result,start=start,stop=stop,update_interval=update_interval)
    end
end