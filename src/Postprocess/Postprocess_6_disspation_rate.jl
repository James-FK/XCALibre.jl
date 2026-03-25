export DissipationRate
@kwdef struct DissipationRate{T<:AbstractScalarField,T1<:AbstractTensorField,T2,V<:AbstractString}
    ϵ::T 
    GradU_mean::T1
    GradU2::T2
    GradU2_mean::T2
    name::V
    start::Union{Real,Nothing}
    stop::Union{Real,Nothing}
    update_interval::Union{Real,Nothing}
end  
function DissipationRate(inputfield; name::String =  "Dissipation_rate", start::Union{Real,Nothing}=nothing, stop::Union{Real,Nothing}=nothing,update_interval::Union{Real,Nothing}=nothing)
    if inputfield isa VectorField
        storage = ScalarField(inputfield.mesh)
        GradU_mean = TensorField(inputfield.mesh)
        GradU2 = ScalarField(inputfield.mesh)
        GradU2_mean = ScalarField(inputfield.mesh)
    else
        throw(ArgumentError("Unsupported field type: $(typeof(inputfield))"))
    end
    return  DissipationRate(ϵ=storage;GradU_mean = GradU_mean,GradU2=GradU2, GradU2_mean = GradU2_mean, name=name, start=start, stop=stop, update_interval=update_interval)
end

function runtime_postprocessing!(DR::DissipationRate{T,T1,T2,V},iter::Integer,n_iterations::Integer,config,S,model,time) where {T<:ScalarField,V,T1,T2}
    if must_calculate(DR,iter,n_iterations)
        n = div(iter - DR.start,DR.update_interval) + 1
        gradU = S.gradU.result
        #I need to calculate the mean of gradU squared and the mean of gradU 
        magnitude2!(DR.GradU2, gradU, config) #current value of gradU squared store in DR.GradU2

        #update running mean of gradU squared, this is a scalarfield so only need to do once
        _update_running_mean!(DR.GradU2_mean.values, DR.GradU2.values,n)

        
        #now need to update the running mean of gradU, this had nine components so needs to be done for all components 
        _update_running_mean!(DR.GradU_mean.xx.values, gradU.xx.values,n)
        _update_running_mean!(DR.GradU_mean.xy.values, gradU.xy.values,n)
        _update_running_mean!(DR.GradU_mean.xz.values, gradU.xz.values,n)
        _update_running_mean!(DR.GradU_mean.yx.values, gradU.yx.values,n)
        _update_running_mean!(DR.GradU_mean.yy.values, gradU.yy.values,n)
        _update_running_mean!(DR.GradU_mean.yz.values, gradU.yz.values,n)
        _update_running_mean!(DR.GradU_mean.zx.values, gradU.zx.values,n)
        _update_running_mean!(DR.GradU_mean.zy.values, gradU.zy.values,n)
        _update_running_mean!(DR.GradU_mean.zz.values, gradU.zz.values,n)

        #now calculate the dissipation rate and store in ϵ
        magnitude2!(DR.ϵ, DR.GradU_mean, config; scale_factor = -1.0) # this calculates -1 * the magnitude of of time averaged gradU

        #finally ϵ is the sum of mean(gradU²) - mean(gradU)²
        @. DR.ϵ.values = DR.ϵ.values + DR.GradU2_mean.values
        

    end
    return nothing
end


function convert_time_to_iterations(DR::DissipationRate, model,dt,iterations)
    if model.time === Transient()
        if DR.start === nothing
            start = 1
        else 
            DR.start >= 0  || throw(ArgumentError("Start must be a value ≥ 0 (got $(DR.start))"))
            start = clamp(ceil(Int, DR.start / dt), 1, iterations) 
        end

        if DR.stop === nothing 
            stop = iterations
        else
            DR.stop ≥ 0 || throw(ArgumentError("stop must be ≥ 0 (got $(DR.stop))"))
            stop = clamp(floor(Int,DR.stop / dt), 1, iterations)
        end

        if DR.update_interval === nothing 
            update_interval = 1
        else
            DR.update_interval > 0 || throw(ArgumentError("update interval must be > 0 (got $(DR.update_interval))"))
            update_interval = max(1, floor(Int,DR.update_interval / dt))
        end
        stop >= start || throw(ArgumentError("After conversion with dt=$dt the averaging window is empty (start = $start, stop = $stop)"))
        return DissipationRate(ϵ=DR.ϵ;GradU_mean = DR.GradU_mean,GradU2=DR.GradU2, GradU2_mean = DR.GradU2_mean, name=DR.name, start=start, stop=stop, update_interval=update_interval)

    else #for Steady runs use iterations 
        if DR.start === nothing
            start = 1
        else 
            DR.start isa Integer || throw(ArgumentError("For steady runs, start must be specified in iterations and therefore be an integer (got $(DR.start))"))
            DR.start >=1     || throw(ArgumentError("Start must be ≥1 (got $(DR.start))"))
            start = DR.start
        end

        if DR.stop === nothing 
            stop = iterations
        else
            DR.stop isa Integer || throw(ArgumentError("For steady runs, stop must be specified in iterations and therefore be an integer (got $(DR.stop))"))
            DR.stop >=1     || throw(ArgumentError("Stop must be ≥1 (got $(DR.stop))"))
            stop = DR.stop
        end

        if DR.update_interval === nothing 
            update_interval = 1
        else
            DR.update_interval isa Integer || throw(ArgumentError("For steady runs, update_interval must be specified in iterations and therefore be an integer (got $(DR.update_interval))"))
            DR.update_interval >= 1 || throw(ArgumentError("update interval must be ≥1 (got $(DR.update_interval))"))
            update_interval = DR.update_interval
        end

        stop >= start || throw(ArgumentError("stop iteration needs to be ≥ start  (got start = $start, stop = $stop)"))
        stop <= iterations || throw(ArgumentError("stop ($stop) must be ≤ iterations ($iterations)"))
        return DissipationRate(ϵ=DR.ϵ;GradU_mean = DR.GradU_mean,GradU2=DR.GradU2, GradU2_mean = DR.GradU2_mean, name=DR.name, start=start, stop=stop, update_interval=update_interval)
    end
end