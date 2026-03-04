export runtime_postprocessing!
export FieldAverageWSS
export convert_time_to_iterations
@kwdef struct FieldAverageWSS{T<:AbstractField,V<:AbstractVector,P<:Symbol,S<:AbstractString}
    tauw::T
    pos::V
    patch::P
    name::S
    start::Union{Real,Nothing}
    stop::Union{Real,Nothing}
    update_interval::Union{Real,Nothing}
end

function FieldAverageWSS(model;patch::Symbol,name::AbstractString = "Wall_shear_stress", start::Union{Real,Nothing}=nothing, stop::Union{Real,Nothing}=nothing,update_interval::Union{Real,Nothing}=nothing)
    #create storage of the appropriate length 
    mesh  = model.domain
    boundaries = mesh.boundaries
    ID = boundary_index(boundaries, patch)
    boundary = boundaries[ID]
    (; IDs_range) = boundary
    pos = fill(SVector{3,Float64}(0,0,0), length(IDs_range))
    x = FaceScalarField(zeros(Float64, length(IDs_range)), mesh)
    y = FaceScalarField(zeros(Float64, length(IDs_range)), mesh)
    z = FaceScalarField(zeros(Float64, length(IDs_range)), mesh)
    tauw = FaceVectorField(x,y,z, mesh)

    return FieldAverageWSS(tauw=tauw,pos=pos, patch=patch, name=name,start=start, stop=stop, update_interval=update_interval)
end

function runtime_postprocessing!(avg::FieldAverageWSS{T,V,P,S},iter::Integer,n_iterations::Integer,config,Str,model) where {T,V,P,S}
    if must_calculate(avg,iter,n_iterations)
        n = div(iter - avg.start,avg.update_interval) + 1
        current_tauw, pos = wall_shear_stress(avg.patch, model,config) 
        _update_running_mean!(avg.tauw.x.values,current_tauw.x.values,n)
        _update_running_mean!(avg.tauw.y.values,current_tauw.y.values,n)
        _update_running_mean!(avg.tauw.z.values,current_tauw.z.values,n)
    end
    if iter == n_iterations
        shear = avg.tauw
        pos = wall_shear_stress(avg.patch, model,config)[2]
        open("ShearStress.txt", "w") do io
        for (i, p) in enumerate(pos)
            println(io,
                p[1], ' ', p[2], ' ', p[3], ' ',
                shear.x.values[i], ' ', shear.y.values[i], ' ', shear.z.values[i]
            )
        end
end
    end
    return nothing
end


function convert_time_to_iterations(avg::FieldAverageWSS, model,dt,iterations)
    if model.time === Transient()
        if avg.start === nothing
            start = 1
        else 
            avg.start >= 0  || throw(ArgumentError("Start must be a value ≥ 0 (got $(avg.start))"))
            start = clamp(ceil(Int, avg.start / dt), 1, iterations) 
        end

        if avg.stop === nothing 
            stop = iterations
        else
            avg.stop ≥ 0 || throw(ArgumentError("stop must be ≥ 0 (got $(avg.stop))"))
            stop = clamp(floor(Int,avg.stop / dt), 1, iterations)
        end

        if avg.update_interval === nothing 
            update_interval = 1
        else
            avg.update_interval > 0 || throw(ArgumentError("update interval must be > 0 (got $(avg.update_interval))"))
            update_interval = max(1, floor(Int,avg.update_interval / dt))
        end
        stop >= start || throw(ArgumentError("After conversion with dt=$dt the averaging window is empty (start = $start, stop = $stop)"))
        return FieldAverageWSS(tauw=avg.tauw,pos=avg.pos, patch=avg.patch, name=avg.name,start=start, stop=stop, update_interval=update_interval)

    else #for Steady runs use iterations 
        if avg.start === nothing
            start = 1
        else 
            avg.start isa Integer || throw(ArgumentError("For steady runs, start must be specified in iterations and therefore be an integer (got $(avg.start))"))
            avg.start >=1     || throw(ArgumentError("Start must be ≥1 (got $(avg.start))"))
            start = avg.start
        end

        if avg.stop === nothing 
            stop = iterations
        else
            avg.stop isa Integer || throw(ArgumentError("For steady runs, stop must be specified in iterations and therefore be an integer (got $(avg.stop))"))
            avg.stop >=1     || throw(ArgumentError("Stop must be ≥1 (got $(avg.stop))"))
            stop = avg.stop
        end

        if avg.update_interval === nothing 
            update_interval = 1
        else
            avg.update_interval isa Integer || throw(ArgumentError("For steady runs, update_interval must be specified in iterations and therefore be an integer (got $(avg.update_interval))"))
            avg.update_interval >= 1 || throw(ArgumentError("update interval must be ≥1 (got $(avg.update_interval))"))
            update_interval = avg.update_interval
        end

        stop >= start || throw(ArgumentError("stop iteration needs to be ≥ start  (got start = $start, stop = $stop)"))
        stop <= iterations || throw(ArgumentError("stop ($stop) must be ≤ iterations ($iterations)"))
        return FieldAverageWSS(tauw=avg.tauw,pos=avg.pos, patch=avg.patch, name=avg.name,start=start, stop=stop, update_interval=update_interval)
    end
end