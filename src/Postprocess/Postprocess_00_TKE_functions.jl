export correct_gradient!
export wall_correction!

correct_gradient!(snGrad, k, kw, wall_ranges, config) = begin
    for IDs_range ∈ wall_ranges
        correct_gradient!(snGrad, k, kw, IDs_range::UnitRange, config)
    end
end

correct_gradient!(snGrad, k, kw, IDs_range::UnitRange, config) = begin
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
    (; normal, delta, e) = face

    kc = k[cID]
    dperp = delta*(e⋅normal) # wall-normal distance (projection of cell-to-face vector)
    grad = (kw - kc)/dperp * normal

    snGrad.x[fID] = grad[1]
    snGrad.y[fID] = grad[2]
    snGrad.z[fID] = grad[3]
    nothing

end


wall_correction!(Kf, wall_ranges, config) = begin
    for IDs_range ∈ wall_ranges
        wall_correction!(Kf, IDs_range::UnitRange, config)
    end
end

wall_correction!(Kf,IDs_range::UnitRange,config) = begin
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

# Face BCs for fluctuation correlations such as ⟨p'u'ⱼ⟩: the U BCs cannot be reused directly
# because a Dirichlet U sets the face value to U itself. Fixed-value U means u' = 0 on that face,
# so the correlation is zero; a time-varying inlet (DirichletFunction) falls back to zero gradient.
fluctuation_boundaries!(psif::FaceVectorField, psi::VectorField, UBCs, time, config) = begin
    correct_boundaries!(psif, psi, UBCs, time, config) # handles periodic, symmetry and outlets
    boundary_cellsID = psif.mesh.boundary_cellsID
    for BC ∈ UBCs
        r = BC.IDs_range
        if BC isa Union{Dirichlet,Wall}
            @views psif.x.values[r] .= 0.0
            @views psif.y.values[r] .= 0.0
            @views psif.z.values[r] .= 0.0
        elseif BC isa DirichletFunction
            cIDs = @view boundary_cellsID[r]
            @views psif.x.values[r] .= psi.x.values[cIDs]
            @views psif.y.values[r] .= psi.y.values[cIDs]
            @views psif.z.values[r] .= psi.z.values[cIDs]
        end
    end
end
