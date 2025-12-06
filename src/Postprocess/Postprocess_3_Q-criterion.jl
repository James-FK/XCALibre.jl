export QCriterion
@kwdef struct QCriterion{T<:AbstractScalarField,S<:AbstractString}
    field::T 
    name::S
    start::Union{Real,Nothing}
    stop::Union{Real,Nothing}
    update_interval::Union{Real,Nothing}
end  

function QCriterion(field; name::String =  "Q-Criterion", start::Union{Real,Nothing}=nothing, stop::Union{Real,Nothing}=nothing,update_interval::Union{Real,Nothing}=nothing)
    if field isa VectorField
        storage = AbstractScalarField(field.mesh)
    else
        throw(ArgumentError("Unsupported field type: $(typeof(field))"))
    end
    return  QCriterion(field=storage;name=name, start=start, stop=stop, update_interval=update_interval)
end

function runtime_postprocessing!(RS::ReynoldsStress{T,T2,S},iter::Integer,n_iterations::Integer) where {T<:VectorField,T2<:SymmetricTensorField,S}
   
    
    return nothing
end
function convert_time_to_iterations(RS::ReynoldsStress, model,dt,iterations)
    if model.time === Transient()
        if RS.start === nothing
            start = 1
        else a
            RS.start >= 0  || throw(ArgumentError("Start must be a value ≥ 0 (got $(RS.start))"))
            start = clamp(ceil(Int, RS.start / dt), 1, iterations) 
        end

        if RS.stop === nothing 
            stop = iterations
        else
            RS.stop ≥ 0 || throw(ArgumentError("stop must be ≥ 0 (got $(RS.stop))"))
            stop = clamp(floor(Int,RS.stop / dt), 1, iterations)
        end

        if RS.update_interval === nothing 
            update_interval = 1
        else
            RS.update_interval > 0 || throw(ArgumentError("update interval must be > 0 (got $(RS.update_interval))"))
            update_interval = max(1, floor(Int,RS.update_interval / dt))
        end
        stop >= start || throw(ArgumentError("After conversion with dt=$dt the averaging window is empty (start = $start, stop = $stop)"))
        return ReynoldsStress(field=RS.field,name=RS.name,mean=RS.mean,mean_sq=RS.mean_sq,rs = RS.rs, start=start,stop=stop,update_interval=update_interval)

    else #for Steady runs use iterations 
        if RS.start === nothing
            start = 1
        else 
            RS.start isa Integer || throw(ArgumentError("For steady runs, start must be specified in iterations and therefore be an integer (got $(RS.start))"))
            RS.start >=1     || throw(ArgumentError("Start must be ≥1 (got $(RS.start))"))
            start = RS.start
        end

        if RS.stop === nothing 
            stop = iterations
        else
            RS.stop isa Integer || throw(ArgumentError("For steady runs, stop must be specified in iterations and therefore be an integer (got $(RS.stop))"))
            RS.stop >=1     || throw(ArgumentError("Stop must be ≥1 (got $(RS.stop))"))
            stop = RS.stop
        end

        if RS.update_interval === nothing 
            update_interval = 1
        else
            RS.update_interval isa Integer || throw(ArgumentError("For steady runs, update_interval must be specified in iterations and therefore be an integer (got $(RS.update_interval))"))
            RS.update_interval >= 1 || throw(ArgumentError("update interval must be ≥1 (got $(RS.update_interval))"))
            update_interval = RS.update_interval
        end

        stop >= start || throw(ArgumentError("stop iteration needs to be ≥ start  (got start = $start, stop = $stop)"))
        stop <= iterations || throw(ArgumentError("stop ($stop) must be ≤ iterations ($iterations)"))
        return ReynoldsStress(field=RS.field,name=RS.name,mean=RS.mean,mean_sq=RS.mean_sq,rs = RS.rs, start=start,stop=stop,update_interval=update_interval)
    end
end
