export Probe
@kwdef struct Probe{T<:AbstractField,I<:Integer,S<:AbstractString}
    field::T
    index::I
    name::S
    start::Union{Real,Nothing}
    stop::Union{Real,Nothing}
    update_interval::Union{Real,Nothing}
    write_interval::Union{Real,Nothing}
end

function Probe(field,mesh_cpu;location::AbstractVector, name::AbstractString,start::Union{Real,Nothing}=nothing,stop::Union{Real,Nothing}=nothing,update_interval::Union{Real,Nothing}=nothing, write_interval::Union{Real,Nothing}=nothing)
    index, best_centre = find_nearest_cell_index(mesh_cpu,location)
    @info "Nearest cell centre located at $best_centre "

    return Probe(field=field, index=index,name=name,start=start,stop=stop,update_interval=update_interval,write_interval=write_interval)
end

function runtime_postprocessing!(prb::Probe{T,I,S},iter::Integer,n_iterations::Integer) where {T<:ScalarField,I,S}
    if must_calculate(prb,iter,n_iterations)
        index = prb.index
        current_value = prb.field.values[index]
        write_probe_to_txt(time,current_value,name)
    end
    return nothing
end

#just write a function that writes out the time and the value of the scalar/vector field at that instant 
# function write_probe_to_txt(time,current_value,name)
#     open("ShearStress.txt", "w") do io
#     for (i, p) in enumerate(pos)
#         println(io,
#             p[1], ' ', p[2], ' ', p[3], ' ', shear.x.values[i], ' ', shear.y.values[i], ' ', shear.z.values[i]
#             )
#     end
# end




function find_nearest_cell_index(mesh, vector_coords)
    best_index = 1
    best_distance = Inf

    @inbounds for i ∈ eachindex(mesh.cells)
        ctr = mesh.cells[i].centre
        distance =  (ctr[1] - vector_coords[1])^2 + (ctr[2] - vector_coords[2])^2 + (ctr[3] - vector_coords[3])^2
        if distance < best_distance
            best_distance = distance
            best_index = i
        end
    end
    best_centre = mesh.cells[best_index].centre

    return best_index, best_centre
end