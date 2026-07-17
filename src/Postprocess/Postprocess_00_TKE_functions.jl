export correct_gradient!
export wall_correction!

correct_gradient!(snGrad, k, kw, IDs_range,config) = begin
    mesh = k.mesh
    (;hardware) = config
    (;backend, workgroup) = hardware
    (; faces, boundary_cellsID) = mesh
    ndrange = length(IDs_range)
    kernel! = _correct_gradient!(_setup(backend, workgroup, ndrange)...)
    kernel!(snGrad,k,kw,IDs_range,faces,boundary_cellsID)
end

@kernel function _correct_gradient!(snGrad,k,kw,IDs_range,faces,boundary_cellsID)
    i = @index(Global)
    fID = IDs_range[i]
    cID = boundary_cellsID[fID]

    face = faces[fID]
    (; normal, delta) = face

    kc = k[cID]
    grad = (kw - kc)/delta * normal

    snGrad.x[fID] = grad[1]
    snGrad.y[fID] = grad[2]
    snGrad.z[fID] = grad[3] 
    nothing   

end


wall_correction!(Kf,IDs_range,config) = begin
    (;hardware) = config
    (;backend, workgroup) = hardware
    ndrange = length(IDs_range)
    kernel! = _wall_correction!(_setup(backend, workgroup, ndrange)...)
    kernel!(Kf.values,IDs_range)
end
@kernel function _wall_correction!(Kf_values,IDs_range)
    i = @index(Global)
    fID = IDs_range[i]
    Kf_values[fID] = 0.0
    nothing   
end