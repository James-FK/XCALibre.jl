export TKEBudget

@kwdef struct TKEBudget{V<:AbstractVector,TU,Tp,Tnu,Tnut,VF,SF,TF,ST,FS,FV,G}
    names::V
    # fields of the model being sampled
    U::TU
    p::Tp
    nu::Tnu
    nut::Tnut
    # raw moments, accumulated on every sample
    meanU::VF       # ⟨Uᵢ⟩
    meanUU::ST      # ⟨UᵢUⱼ⟩
    meangradU::TF   # ⟨∂Uᵢ/∂xⱼ⟩
    meangradU2::SF  # ⟨∂Uᵢ/∂xⱼ ∂Uᵢ/∂xⱼ⟩
    meanp::SF       # ⟨p⟩
    meanpU::VF      # ⟨pUⱼ⟩
    meanUiUiUj::VF  # ⟨UᵢUᵢUⱼ⟩
    τmean::ST       # ⟨τᵢⱼ⟩
    meanτgradU::SF  # ⟨τᵢⱼ ∂Uᵢ/∂xⱼ⟩
    meanUτ::VF      # ⟨Uᵢτᵢⱼ⟩
    # work fields, used only when the budget terms are evaluated
    k::SF
    kf::FS
    gradk::G
    flux::VF
    fluxf::FV
    # budget terms
    convection::SF
    production::SF
    diffusion_pressure::SF
    diffusion_turbulent::SF
    diffusion_viscous::SF
    diffusion_SGS::SF
    dissipation::SF
    dissipation_SGS::SF
    start::Union{Real,Nothing}
    stop::Union{Real,Nothing}
    update_interval::Union{Real,Nothing}
end

"""
    TKEBudget(
    #required arguments
    model;

    #optional keyword arguments
    names::Vector,
    start::Union{Real,Nothing},
    stop::Union{Real,Nothing},
    update_interval::Union{Real,Nothing})

Constructor to allocate memory to store the terms of the turbulent kinetic energy (TKE) budget. Once created, should be passed to the `Configuration` object as an argument with keyword `postprocess`

Every term is a right-hand-side contribution to `∂k/∂t`, with `k = ½⟨u'ᵢu'ᵢ⟩`, so in a statistically steady flow the eight terms sum to zero

| field                 | term                                |
|:----------------------|:------------------------------------|
| `convection`          | `-⟨Uⱼ⟩ ∂k/∂xⱼ`                      |
| `production`          | `-⟨u'ᵢu'ⱼ⟩ ∂⟨Uᵢ⟩/∂xⱼ`               |
| `diffusion_pressure`  | `-∂⟨p'u'ⱼ⟩/∂xⱼ`                     |
| `diffusion_turbulent` | `-½ ∂⟨u'ᵢu'ᵢu'ⱼ⟩/∂xⱼ`               |
| `diffusion_viscous`   | `ν ∂²k/∂xⱼ∂xⱼ`                      |
| `diffusion_SGS`       | `-∂⟨u'ᵢτ'ᵢⱼ⟩/∂xⱼ`                   |
| `dissipation`         | `-ν⟨∂u'ᵢ/∂xⱼ ∂u'ᵢ/∂xⱼ⟩`             |
| `dissipation_SGS`     | `⟨τ'ᵢⱼ ∂u'ᵢ/∂xⱼ⟩`                   |

where `τᵢⱼ = -2νₜSᵢⱼ` is the modelled (subgrid scale) stress, which is zero for laminar models, and `p` is the kinematic pressure.

# Input arguments
- `model` the `Physics` object of an incompressible simulation. Its velocity, pressure, viscosity and eddy viscosity fields are sampled.

# Optional arguments
- `names::Vector` the names of the eight output fields, in the order of the table above (which is the default).
- `start::Union{Real,Nothing}` optional keyword which specifies the start of the averaging window, for **steady** simulations, this is in **iterations**, for **transient** simulations it is in **flow time**.
- `stop::Union{Real,Nothing}` optional keyword which specifies the end iteration/time of the averaging window. Default value is the last iteration/timestep.
- `update_interval::Union{Real,Nothing}` optional keyword which specifies how often the statistics are sampled (default value is 1 i.e. every timestep/iteration).

Only raw moments are accumulated when sampling. The budget terms are evaluated from them when results are written (`write_interval` in `Runtime`) and on the last iteration, so between writes the term fields hold the values of the last write.
"""
function TKEBudget(model;
    names::Vector = ["convection", "production", "diffusion_pressure", "diffusion_turbulent",
        "diffusion_viscous", "diffusion_SGS", "dissipation", "dissipation_SGS"],
    start::Union{Real,Nothing}=nothing,
    stop::Union{Real,Nothing}=nothing,
    update_interval::Union{Real,Nothing}=nothing)

    length(names) == 8 || throw(ArgumentError("names must hold 8 field names (got $(length(names)))"))
    model.fluid isa AbstractIncompressible || throw(ArgumentError(
        "TKEBudget is only available for incompressible fluids (got $(nameof(typeof(model.fluid))))"))

    mesh = model.domain
    (; U, p) = model.momentum
    nut = model.turbulence isa Laminar ? ConstantScalar(zero(_get_float(mesh))) : model.turbulence.nut

    return TKEBudget(
        names = names,
        U = U,
        p = p,
        nu = model.fluid.nu,
        nut = nut,
        meanU = VectorField(mesh),
        meanUU = SymmetricTensorField(mesh),
        meangradU = TensorField(mesh),
        meangradU2 = ScalarField(mesh),
        meanp = ScalarField(mesh),
        meanpU = VectorField(mesh),
        meanUiUiUj = VectorField(mesh),
        τmean = SymmetricTensorField(mesh),
        meanτgradU = ScalarField(mesh),
        meanUτ = VectorField(mesh),
        k = ScalarField(mesh),
        kf = FaceScalarField(mesh),
        gradk = Grad{Gauss}(ScalarField(mesh)),
        flux = VectorField(mesh),
        fluxf = FaceVectorField(mesh),
        convection = ScalarField(mesh),
        production = ScalarField(mesh),
        diffusion_pressure = ScalarField(mesh),
        diffusion_turbulent = ScalarField(mesh),
        diffusion_viscous = ScalarField(mesh),
        diffusion_SGS = ScalarField(mesh),
        dissipation = ScalarField(mesh),
        dissipation_SGS = ScalarField(mesh),
        start = start,
        stop = stop,
        update_interval = update_interval
    )
end

# Mesh-free storage of a SymmetricTensorField (6 unique components), indexed as a 3x3 matrix
struct SymmetricValues{A}
    xx::A
    xy::A
    xz::A
    yy::A
    yz::A
    zz::A
end
Adapt.@adapt_structure SymmetricValues

_symmetric_values(t::SymmetricTensorField) = SymmetricValues(
    t.xx.values, t.xy.values, t.xz.values, t.yy.values, t.yz.values, t.zz.values)

@inline Base.getindex(t::SymmetricValues, i::Integer) = @inbounds SMatrix{3,3}(
    t.xx[i], t.xy[i], t.xz[i],
    t.xy[i], t.yy[i], t.yz[i],
    t.xz[i], t.yz[i], t.zz[i])

# Running mean update m ← m + a(x - m) with a = 1/n, for scalar, vector and tensor storage
@inline _running_mean!(m, i, a, x) = @inbounds m[i] += a*(x - m[i])

@inline function _running_mean!(m::SymmetricValues, i, a, x)
    @inbounds begin
        m.xx[i] += a*(x[1,1] - m.xx[i])
        m.xy[i] += a*(x[1,2] - m.xy[i])
        m.xz[i] += a*(x[1,3] - m.xz[i])
        m.yy[i] += a*(x[2,2] - m.yy[i])
        m.yz[i] += a*(x[2,3] - m.yz[i])
        m.zz[i] += a*(x[3,3] - m.zz[i])
    end
    nothing
end

@inline function _running_mean!(m::TensorValues, i, a, x)
    @inbounds begin
        m.xx[i] += a*(x[1,1] - m.xx[i])
        m.xy[i] += a*(x[1,2] - m.xy[i])
        m.xz[i] += a*(x[1,3] - m.xz[i])
        m.yx[i] += a*(x[2,1] - m.yx[i])
        m.yy[i] += a*(x[2,2] - m.yy[i])
        m.yz[i] += a*(x[2,3] - m.yz[i])
        m.zx[i] += a*(x[3,1] - m.zx[i])
        m.zy[i] += a*(x[3,2] - m.zy[i])
        m.zz[i] += a*(x[3,3] - m.zz[i])
    end
    nothing
end

# Only raw moments are accumulated every sample. Fluctuation statistics are recovered exactly
# from them, e.g. ⟨∂u'ᵢ/∂xⱼ ∂u'ᵢ/∂xⱼ⟩ = ⟨∂Uᵢ/∂xⱼ ∂Uᵢ/∂xⱼ⟩ - ⟨∂Uᵢ/∂xⱼ⟩⟨∂Uᵢ/∂xⱼ⟩, so the budget terms
# are only evaluated on iterations where they are written out (and on the last iteration).
# Every term is written as a right-hand-side contribution to ∂k/∂t, so for a statistically
# steady flow the residual is the sum of all eight terms
# 0 = C + P + Πₚ + Tₜ + Dᵥ + D_SGS + ϵ + ϵ_SGS
function runtime_postprocessing!(tke::TKEBudget, iter::Integer, n_iterations::Integer, S, time, config)
    if must_calculate(tke, iter, n_iterations)
        n = div(iter - tke.start, tke.update_interval) + 1
        accumulate_moments!(tke, S, n, config)
    end

    write_interval = config.runtime.write_interval
    writing = iter%write_interval + signbit(write_interval) == 0
    if iter >= tke.start && (writing || iter == n_iterations)
        compute_budget_terms!(tke, time, config)
    end
    return nothing
end

function accumulate_moments!(tke::TKEBudget, S, n, config)
    a = one(eltype(tke.meanp.values))/n

    # Bind values, not fields, so the closure carries no mesh (xcal_foreach runs on any backend)
    U, p, nut = field_values(tke.U), field_values(tke.p), field_values(tke.nut)
    gradU = field_values(S.gradU.result)
    meanU, meanUU = field_values(tke.meanU), _symmetric_values(tke.meanUU)
    meangradU, meangradU2 = field_values(tke.meangradU), field_values(tke.meangradU2)
    meanp, meanpU = field_values(tke.meanp), field_values(tke.meanpU)
    meanUiUiUj = field_values(tke.meanUiUiUj)
    τmean, meanτgradU = _symmetric_values(tke.τmean), field_values(tke.meanτgradU)
    meanUτ = field_values(tke.meanUτ)

    xcal_foreach(meanp, config) do i
        @inbounds begin
            Ui = U[i]
            pc = p[i]
            G = gradU[i]
            τ = -nut[i]*(G + G') # modelled stress τᵢⱼ = -2νₜSᵢⱼ

            _running_mean!(meanU, i, a, Ui)
            _running_mean!(meanUU, i, a, Ui*Ui')
            _running_mean!(meangradU, i, a, G)
            _running_mean!(meangradU2, i, a, G⋅G)
            _running_mean!(meanp, i, a, pc)
            _running_mean!(meanpU, i, a, pc*Ui)
            _running_mean!(meanUiUiUj, i, a, (Ui⋅Ui)*Ui)
            _running_mean!(τmean, i, a, τ)
            _running_mean!(meanτgradU, i, a, τ⋅G)
            _running_mean!(meanUτ, i, a, τ*Ui) # τ is symmetric: Uᵢτᵢⱼ = τⱼᵢUᵢ
        end
    end
    return nothing
end

function compute_budget_terms!(tke::TKEBudget, time, config)
    mesh = tke.k.mesh
    UBCs = config.boundaries.U
    (; backend) = config.hardware
    (; flux) = tke

    nu = field_values(tke.nu)
    meanU, meanUU = field_values(tke.meanU), _symmetric_values(tke.meanUU)
    meangradU, meangradU2 = field_values(tke.meangradU), field_values(tke.meangradU2)
    meanp, meanpU = field_values(tke.meanp), field_values(tke.meanpU)
    meanUiUiUj = field_values(tke.meanUiUiUj)
    τmean, meanτgradU = _symmetric_values(tke.τmean), field_values(tke.meanτgradU)
    meanUτ = field_values(tke.meanUτ)
    k, fluxv = field_values(tke.k), field_values(flux)
    production, dissipation = field_values(tke.production), field_values(tke.dissipation)
    dissipation_SGS = field_values(tke.dissipation_SGS)

    ###### Terms without spatial derivatives of the statistics ######

    xcal_foreach(k, config) do i
        @inbounds begin
            Um = meanU[i]
            G = meangradU[i]
            R = meanUU[i] - Um*Um' # Reynolds stress Rᵢⱼ = ⟨UᵢUⱼ⟩ - ⟨Uᵢ⟩⟨Uⱼ⟩

            # Production = -⟨u'ᵢu'ⱼ⟩ ∂⟨Uᵢ⟩/∂xⱼ
            production[i] = -(R⋅G)

            # Dissipation = -ν⟨∂u'ᵢ/∂xⱼ ∂u'ᵢ/∂xⱼ⟩ = -ν(⟨∂Uᵢ/∂xⱼ ∂Uᵢ/∂xⱼ⟩ - ⟨∂Uᵢ/∂xⱼ⟩⟨∂Uᵢ/∂xⱼ⟩)
            dissipation[i] = -nu[i]*(meangradU2[i] - G⋅G)

            # ϵ_SGS = ⟨τ'ᵢⱼ ∂u'ᵢ/∂xⱼ⟩ = ⟨τᵢⱼ ∂Uᵢ/∂xⱼ⟩ - ⟨τᵢⱼ⟩⟨∂Uᵢ/∂xⱼ⟩ (a sink, like the viscous dissipation)
            dissipation_SGS[i] = meanτgradU[i] - τmean[i]⋅G

            k[i] = tr(R)/2
        end
    end

    ###### Diffusion terms: divergence of a flux of fluctuation correlations ######

    ## Diffusion due to pressure = -∂⟨p'u'ⱼ⟩/∂xⱼ, with ⟨p'u'ⱼ⟩ = ⟨pUⱼ⟩ - ⟨p⟩⟨Uⱼ⟩ ##
    xcal_foreach(k, config) do i
        @inbounds fluxv[i] = meanpU[i] - meanp[i]*meanU[i]
    end
    fluctuation_divergence!(tke.diffusion_pressure, tke, -1, UBCs, time, config)

    ## Diffusion due to fluctuations = -½ ∂⟨u'ᵢu'ᵢu'ⱼ⟩/∂xⱼ ##
    # ⟨u'ᵢu'ᵢu'ⱼ⟩ = ⟨UᵢUᵢUⱼ⟩ - ⟨Uⱼ⟩⟨UᵢUᵢ⟩ - 2⟨Uᵢ⟩⟨UᵢUⱼ⟩ + 2⟨Uᵢ⟩⟨Uᵢ⟩⟨Uⱼ⟩
    xcal_foreach(k, config) do i
        @inbounds begin
            Um = meanU[i]
            UU = meanUU[i]
            fluxv[i] = meanUiUiUj[i] - tr(UU)*Um - 2*(UU*Um) + 2*(Um⋅Um)*Um
        end
    end
    fluctuation_divergence!(tke.diffusion_turbulent, tke, -1/2, UBCs, time, config)

    ## Subgrid scale diffusion = -∂⟨u'ᵢτ'ᵢⱼ⟩/∂xⱼ, with ⟨u'ᵢτ'ᵢⱼ⟩ = ⟨Uᵢτᵢⱼ⟩ - ⟨Uᵢ⟩⟨τᵢⱼ⟩ ##
    xcal_foreach(k, config) do i
        @inbounds fluxv[i] = meanUτ[i] - τmean[i]*meanU[i]
    end
    fluctuation_divergence!(tke.diffusion_SGS, tke, -1, UBCs, time, config)

    ###### Terms with gradients of k ######

    periodicBCs = filter(BC -> BC isa Union{Periodic,PeriodicParent}, UBCs)
    walls = [BC.IDs_range for BC ∈ UBCs if BC isa Wall && !isempty(BC.IDs_range)]

    # ∇k from face values: k = 0 on walls, zero gradient on every other (non-periodic) boundary
    sync!(tke.k, mesh, config) # no-op serial
    interpolate!(tke.kf, tke.k, config)
    isempty(periodicBCs) || correct_boundaries!(tke.kf, tke.k, periodicBCs, time, config)
    foreach(IDs_range -> wall_correction!(tke.kf, IDs_range, config), walls)
    green_gauss!(tke.gradk, tke.kf, config)
    gradk = tke.gradk.result
    sync!(gradk, mesh, config) # ghost cells have incomplete face lists

    ## Diffusion due to viscosity = ν∇²k ##
    interpolate!(tke.fluxf, gradk, config)
    isempty(periodicBCs) || correct_boundaries!(tke.fluxf, gradk, periodicBCs, time, config)
    foreach(IDs_range -> correct_gradient!(tke.fluxf, tke.k, IDs_range, config), walls)
    div!(tke.diffusion_viscous, tke.fluxf, config)
    diffusion_viscous, gradkv = field_values(tke.diffusion_viscous), field_values(gradk)
    convection = field_values(tke.convection)
    xcal_foreach(k, config) do i
        @inbounds begin
            diffusion_viscous[i] *= nu[i]

            ## Convection = -⟨Uⱼ⟩∂k/∂xⱼ ##
            convection[i] = -(meanU[i]⋅gradkv[i])
        end
    end

    KernelAbstractions.synchronize(backend)
    return nothing
end

# out = scale ∇⋅flux for the cell flux of fluctuation correlations held in tke.flux
function fluctuation_divergence!(out, tke::TKEBudget, scale, UBCs, time, config)
    (; flux, fluxf) = tke
    # ghost values feed the faces shared with other ranks: the SGS flux depends on gradU, whose
    # ghost values are not exchanged (no-op serial)
    sync!(flux, flux.mesh, config)
    interpolate!(fluxf, flux, config)
    fluctuation_boundaries!(fluxf, flux, UBCs, time, config)
    div!(out, fluxf, config)
    out.values .*= eltype(out.values)(scale)
    return nothing
end

# Face BCs for fluctuation correlations such as ⟨p'u'ⱼ⟩: the U BCs cannot be reused directly
# because a Dirichlet U sets the face value to U itself. Fixed-value U means u' = 0 on that face,
# so the correlation is zero; a time-varying inlet (DirichletFunction) falls back to zero gradient.
function fluctuation_boundaries!(psif::FaceVectorField, psi::VectorField, UBCs, time, config)
    correct_boundaries!(psif, psi, UBCs, time, config) # handles periodic, symmetry and outlets
    for BC ∈ UBCs
        fluctuation_boundary!(BC, psif, psi, config)
    end
    return nothing
end

fluctuation_boundary!(BC, psif, psi, config) = nothing

fluctuation_boundary!(BC::Union{Dirichlet,Wall,RotatingWall}, psif, psi, config) = begin
    (; IDs_range) = BC
    isempty(IDs_range) && return nothing
    (; backend, workgroup) = config.hardware
    kernel! = _sized(_zero_patch!, backend, workgroup, length(IDs_range))
    kernel!(psif.x.values, psif.y.values, psif.z.values, IDs_range)
    nothing
end

fluctuation_boundary!(BC::DirichletFunction, psif, psi, config) = begin
    (; IDs_range) = BC
    isempty(IDs_range) && return nothing
    (; backend, workgroup) = config.hardware
    kernel! = _sized(_zero_gradient_patch!, backend, workgroup, length(IDs_range))
    kernel!(psif.x.values, psif.y.values, psif.z.values, field_values(psi), IDs_range,
        psif.mesh.boundary_cellsID)
    nothing
end

@kernel inbounds=true function _zero_patch!(x, y, z, IDs_range)
    i = @index(Global)
    fID = IDs_range[i]
    x[fID] = zero(eltype(x))
    y[fID] = zero(eltype(y))
    z[fID] = zero(eltype(z))
end

@kernel inbounds=true function _zero_gradient_patch!(x, y, z, psi, IDs_range, boundary_cellsID)
    i = @index(Global)
    fID = IDs_range[i]
    psic = psi[boundary_cellsID[fID]]
    x[fID] = psic[1]
    y[fID] = psic[2]
    z[fID] = psic[3]
end

# k = 0 on the faces of a wall patch
wall_correction!(kf::FaceScalarField, IDs_range::UnitRange, config) = begin
    (; backend, workgroup) = config.hardware
    kernel! = _sized(_wall_correction!, backend, workgroup, length(IDs_range))
    kernel!(kf.values, IDs_range)
    nothing
end

@kernel inbounds=true function _wall_correction!(kf, IDs_range)
    i = @index(Global)
    kf[IDs_range[i]] = zero(eltype(kf))
end

# Face gradient of k on a wall patch: one-sided normal gradient to the wall value k = 0
correct_gradient!(gradkf::FaceVectorField, k::ScalarField, IDs_range::UnitRange, config) = begin
    (; faces, boundary_cellsID) = k.mesh
    (; backend, workgroup) = config.hardware
    kernel! = _sized(_correct_gradient!, backend, workgroup, length(IDs_range))
    kernel!(gradkf.x.values, gradkf.y.values, gradkf.z.values, k.values, IDs_range, faces,
        boundary_cellsID)
    nothing
end

@kernel inbounds=true function _correct_gradient!(x, y, z, k, IDs_range, faces, boundary_cellsID)
    i = @index(Global)
    fID = IDs_range[i]
    cID = boundary_cellsID[fID]
    # the boundary face delta is the wall-normal distance of the cell centre
    grad = -k[cID]/faces.delta[fID]*faces.normal[fID]
    x[fID] = grad[1]
    y[fID] = grad[2]
    z[fID] = grad[3]
end

function convert_time_to_iterations(tke::TKEBudget, model, dt, iterations)

    if model.time === Transient()

        if tke.start === nothing
            start = 1
        else
            tke.start >= 0 || throw(ArgumentError("Start must be ≥ 0 (got $(tke.start))"))
            start = clamp(ceil(Int, tke.start / dt), 1, iterations)
        end

        if tke.stop === nothing
            stop = iterations
        else
            tke.stop >= 0 || throw(ArgumentError("Stop must be ≥ 0 (got $(tke.stop))"))
            stop = clamp(floor(Int, tke.stop / dt), 1, iterations)
        end

        if tke.update_interval === nothing
            update_interval = 1
        else
            tke.update_interval > 0 || throw(ArgumentError("update_interval must be > 0 (got $(tke.update_interval))"))
            update_interval = max(1, floor(Int, tke.update_interval / dt))
        end

        stop >= start || throw(ArgumentError("After conversion with dt=$dt the averaging window is empty (start=$start, stop=$stop)"))

    else  # steady

        if tke.start === nothing
            start = 1
        else
            tke.start isa Integer || throw(ArgumentError("For steady runs, start must be an integer (got $(tke.start))"))
            tke.start >= 1 || throw(ArgumentError("Start must be ≥ 1 (got $(tke.start))"))
            start = tke.start
        end

        if tke.stop === nothing
            stop = iterations
        else
            tke.stop isa Integer || throw(ArgumentError("For steady runs, stop must be an integer (got $(tke.stop))"))
            tke.stop >= 1 || throw(ArgumentError("Stop must be ≥ 1 (got $(tke.stop))"))
            stop = tke.stop
        end

        if tke.update_interval === nothing
            update_interval = 1
        else
            tke.update_interval isa Integer || throw(ArgumentError("For steady runs, update_interval must be an integer (got $(tke.update_interval))"))
            tke.update_interval >= 1 || throw(ArgumentError("update_interval must be ≥ 1 (got $(tke.update_interval))"))
            update_interval = tke.update_interval
        end

        stop >= start || throw(ArgumentError("stop must be ≥ start (got start=$start, stop=$stop)"))
        stop <= iterations || throw(ArgumentError("stop ($stop) must be ≤ iterations ($iterations)"))
    end

    # copy every field across, replacing only the converted averaging window
    fields = fieldnames(TKEBudget)
    kwargs = merge(NamedTuple{fields}(getfield.(Ref(tke), fields)),
        (start = start, stop = stop, update_interval = update_interval))
    return TKEBudget(; kwargs...)
end
