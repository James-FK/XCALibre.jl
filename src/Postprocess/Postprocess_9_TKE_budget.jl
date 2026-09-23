export TKEBudget

@kwdef struct TKEBudget{V<:AbstractVector, T1<:AbstractVectorField, T2<:AbstractField, T3<:AbstractScalarField,
    T4<:AbstractField, T5<:AbstractField,  T7<:AbstractField, T8<:AbstractField}
    names::V
    wall_IDs_range::Vector{UnitRange{Int64}}
    meanU::T1
    meanUU::T2
    rst::T2
    meangradU::T4
    meangradU2::T3
    meanp::T3
    meanpU::T1
    meanpUfluc::T1
    meanpUflucf::T5
    meanUiUiUj::T1
    meanuiuiuj::T1
    meanuiuiujf::T5
    k::T3
    kf::T7
    gradk::Grad{<:Any}
    gradkf::T8
    τ::T2
    τmean::T2
    meanτgradU::T3
    Uτ::T1
    meanUτ::T1
    meanUflucτfluc::T1
    Uflucτflucf::T5
    scratch::T3
    convection::T3
    production::T3
    diffusion_pressure::T3
    diffusion_turbulent::T3
    diffusion_viscous::T3
    diffusion_SGS::T3
    dissipation::T3
    dissipation_SGS::T3
    start::Union{Real,Nothing}
    stop::Union{Real,Nothing}
    update_interval::Union{Real,Nothing}
end
function TKEBudget(field,BCs;
    names::Vector = ["convection","production", "diffusion_pressure", "diffusion_turbulent","diffusion_viscous","diffusion_SGS", "dissipation", "dissipation_SGS"],
    start::Union{Real,Nothing}=nothing,
    stop::Union{Real,Nothing}=nothing,
    update_interval::Union{Real,Nothing}=nothing)
    wall_IDs_range = UnitRange{Int64}[] # every Wall patch (e.g. both walls of a channel)
    UBCs = BCs.U
    for b in UBCs
        if b isa Wall
            push!(wall_IDs_range, b.IDs_range)
        end
    end

    if field isa VectorField
        mesh = field.mesh
        return TKEBudget(
            names = names,
            wall_IDs_range = wall_IDs_range,
            meanU = VectorField(mesh),
            meanUU = SymmetricTensorField(mesh),
            rst = SymmetricTensorField(mesh),
            meangradU = TensorField(mesh),
            meangradU2 = ScalarField(mesh),
            meanp = ScalarField(mesh),
            meanpU = VectorField(mesh),
            meanpUfluc = VectorField(mesh),
            meanpUflucf = FaceVectorField(mesh),
            meanUiUiUj = VectorField(mesh),
            meanuiuiuj = VectorField(mesh),
            meanuiuiujf = FaceVectorField(mesh),
            k = ScalarField(mesh),
            kf = FaceScalarField(mesh),
            gradk = Grad{Gauss}(ScalarField(mesh)),
            gradkf = FaceVectorField(mesh),
            τ = SymmetricTensorField(mesh),
            τmean = SymmetricTensorField(mesh),
            meanτgradU = ScalarField(mesh),
            Uτ = VectorField(mesh),
            meanUτ = VectorField(mesh),
            meanUflucτfluc = VectorField(mesh),
            Uflucτflucf = FaceVectorField(mesh),
            scratch = ScalarField(mesh),
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
    else
        throw(ArgumentError("Unsupported field type: $(typeof(field))"))
    end
end


# Only raw moments are accumulated every sample. Fluctuation statistics are recovered exactly
# from them, e.g. ⟨∂u'ᵢ/∂xⱼ ∂u'ᵢ/∂xⱼ⟩ = ⟨∂Uᵢ/∂xⱼ ∂Uᵢ/∂xⱼ⟩ - ⟨∂Uᵢ/∂xⱼ⟩⟨∂Uᵢ/∂xⱼ⟩, so the budget terms
# are only evaluated on iterations where they are written out.
# Every term is written as a right-hand-side contribution to ∂k/∂t, so for a statistically
# steady flow the residual is the sum of all eight terms
# 0 = C + P + Πₚ + Tₜ + Dᵥ + D_SGS + ϵ + ϵ_SGS
function runtime_postprocessing!(tke::TKEBudget,iter::Integer,n_iterations::Integer,config,S,model,time)
    if must_calculate(tke,iter,n_iterations)
        n = div(iter - tke.start,tke.update_interval) + 1
        a = 1.0 / n
        b = 1.0 - a
        U = model.momentum.U
        p = model.momentum.p
        nut = model.turbulence.nut
        gradU = S.gradU.result
        Ux, Uy, Uz = U.x.values, U.y.values, U.z.values
        pv = p.values
        (; scratch) = tke

        _update_running_mean!(tke.meanU, U, n) # ⟨Uᵢ⟩
        _update_running_mean!(tke.meanUU, U, n) # ⟨UᵢUⱼ⟩
        _update_running_mean!(tke.meangradU, gradU, n) # ⟨∂Uᵢ/∂xⱼ⟩
        _update_running_mean!(tke.meanp, p, n) # ⟨p⟩

        # ⟨pUⱼ⟩
        @. tke.meanpU.x.values = b * tke.meanpU.x.values + a * pv * Ux
        @. tke.meanpU.y.values = b * tke.meanpU.y.values + a * pv * Uy
        @. tke.meanpU.z.values = b * tke.meanpU.z.values + a * pv * Uz

        # ⟨UᵢUᵢUⱼ⟩
        @. scratch.values = Ux^2 + Uy^2 + Uz^2
        @. tke.meanUiUiUj.x.values = b * tke.meanUiUiUj.x.values + a * scratch.values * Ux
        @. tke.meanUiUiUj.y.values = b * tke.meanUiUiUj.y.values + a * scratch.values * Uy
        @. tke.meanUiUiUj.z.values = b * tke.meanUiUiUj.z.values + a * scratch.values * Uz

        # ⟨∂Uᵢ/∂xⱼ ∂Uᵢ/∂xⱼ⟩
        magnitude2!(scratch, gradU, config)
        _update_running_mean!(tke.meangradU2, scratch, n)

        # SGS stress τᵢⱼ = -2νₜSᵢⱼ and the moments ⟨τᵢⱼ⟩, ⟨τᵢⱼ ∂Uᵢ/∂xⱼ⟩, ⟨Uᵢτᵢⱼ⟩
        elementwise_multiply!(tke.τ,nut,S,config;scale_factor = -2)
        _update_running_mean!(tke.τmean,tke.τ,n)
        double_inner_product!(scratch, tke.τ, gradU, config)
        _update_running_mean!(tke.meanτgradU, scratch, n)
        elementwise_multiply!(T(tke.Uτ),T(U),tke.τ,config)
        _update_running_mean!(tke.meanUτ, tke.Uτ, n)
    end

    write_interval = config.runtime.write_interval
    if iter >= tke.start && iter%write_interval + signbit(write_interval) == 0
        compute_budget_terms!(tke, config, model, time)
    end

    return nothing
end

function compute_budget_terms!(tke::TKEBudget, config, model, time)
    nu = model.fluid.nu.values
    UBCs = config.boundaries.U
    (; meanU, meanUU) = tke

    ###### The production term = − ⟨u'ᵢu'ⱼ⟩⟨∂Uᵢ/∂xⱼ⟩  ######

    # store the Reynolds Stress Tensor Rᵢⱼ = ⟨UᵢUⱼ⟩ - ⟨Uᵢ⟩⟨Uⱼ⟩
    @. tke.rst.xx.values = meanUU.xx.values - meanU.x.values^2
    @. tke.rst.xy.values = meanUU.xy.values - meanU.x.values * meanU.y.values
    @. tke.rst.xz.values = meanUU.xz.values - meanU.x.values * meanU.z.values
    @. tke.rst.yy.values = meanUU.yy.values - meanU.y.values^2
    @. tke.rst.yz.values = meanUU.yz.values - meanU.y.values * meanU.z.values
    @. tke.rst.zz.values = meanUU.zz.values - meanU.z.values^2

    double_inner_product!(tke.production, tke.rst, tke.meangradU,config; scale_factor = -1.0)

    ###### The Dissipation term  = -ν⟨∂u'ᵢ/∂xⱼ ∂u'ᵢ/∂xⱼ⟩ = -ν(⟨∂Uᵢ/∂xⱼ ∂Uᵢ/∂xⱼ⟩ - ⟨∂Uᵢ/∂xⱼ⟩⟨∂Uᵢ/∂xⱼ⟩) ######

    magnitude2!(tke.dissipation, tke.meangradU, config)
    @. tke.dissipation.values = -nu * (tke.meangradU2.values - tke.dissipation.values)

    ###### The Diffusion terms ######

    ## Diffusion due to pressure = -∂⟨p'u'ⱼ⟩/∂xⱼ ##

    @. tke.meanpUfluc.x.values = tke.meanpU.x.values - meanU.x.values * tke.meanp.values
    @. tke.meanpUfluc.y.values = tke.meanpU.y.values - meanU.y.values * tke.meanp.values
    @. tke.meanpUfluc.z.values = tke.meanpU.z.values - meanU.z.values * tke.meanp.values
    interpolate!(tke.meanpUflucf,tke.meanpUfluc,config)
    fluctuation_boundaries!(tke.meanpUflucf,tke.meanpUfluc,UBCs,time,config)
    div!(tke.diffusion_pressure,tke.meanpUflucf,config)
    @. tke.diffusion_pressure.values = -tke.diffusion_pressure.values

    ## Diffusion due to fluctuations = -½ ∂⟨u'ᵢu'ᵢu'ⱼ⟩/∂xⱼ ##

    # ⟨u'ᵢu'ᵢu'ⱼ⟩ = ⟨UᵢUᵢUⱼ⟩ - ⟨Uⱼ⟩⟨UᵢUᵢ⟩ - 2⟨Uᵢ⟩⟨UᵢUⱼ⟩ + 2⟨Uᵢ⟩⟨Uᵢ⟩⟨Uⱼ⟩
    @. tke.meanuiuiuj.x.values = (tke.meanUiUiUj.x.values - (meanU.x.values * (meanUU.xx.values + meanUU.yy.values + meanUU.zz.values))
                                    - 2 * (meanU.x.values * meanUU.xx.values + meanU.y.values * meanUU.yx.values + meanU.z.values * meanUU.zx.values)
                                    + 2 * (meanU.x.values^2 + meanU.y.values^2 + meanU.z.values^2) * meanU.x.values)

    @. tke.meanuiuiuj.y.values = (tke.meanUiUiUj.y.values - (meanU.y.values * (meanUU.xx.values + meanUU.yy.values + meanUU.zz.values))
                                    - 2 * (meanU.x.values * meanUU.xy.values + meanU.y.values * meanUU.yy.values + meanU.z.values * meanUU.zy.values)
                                    + 2 * (meanU.x.values^2 + meanU.y.values^2 + meanU.z.values^2) * meanU.y.values)

    @. tke.meanuiuiuj.z.values = (tke.meanUiUiUj.z.values - (meanU.z.values * (meanUU.xx.values + meanUU.yy.values + meanUU.zz.values))
                                    - 2 * (meanU.x.values * meanUU.xz.values + meanU.y.values * meanUU.yz.values + meanU.z.values * meanUU.zz.values)
                                    + 2 * (meanU.x.values^2 + meanU.y.values^2 + meanU.z.values^2) * meanU.z.values)
    interpolate!(tke.meanuiuiujf,tke.meanuiuiuj,config)
    fluctuation_boundaries!(tke.meanuiuiujf,tke.meanuiuiuj,UBCs,time,config)
    div!(tke.diffusion_turbulent,tke.meanuiuiujf,config)
    @. tke.diffusion_turbulent.values *= -0.5

    ## Diffusion due to viscosity = ν∇²k ##

    # get k from the 1/2 the trace of the reynolds stress tensor
    @. tke.k.values = 0.5 * (tke.rst.xx.values + tke.rst.yy.values + tke.rst.zz.values )
    interpolate!(tke.kf,tke.k,config)
    wall_correction!(tke.kf,tke.wall_IDs_range,config)
    green_gauss!(tke.gradk,tke.kf,config) #calculate gradk

    interpolate!(tke.gradkf,tke.gradk.result,config)
    correct_gradient!(tke.gradkf, tke.k, 0.0, tke.wall_IDs_range, config)
    div!(tke.diffusion_viscous,tke.gradkf,config)
    @. tke.diffusion_viscous.values = tke.diffusion_viscous.values * nu

    ## Subgrid scale contributions to budget in case of LES ##

    # ϵ_SGS = ⟨τ'ᵢⱼ ∂u'ᵢ/∂xⱼ⟩ = ⟨τᵢⱼ ∂Uᵢ/∂xⱼ⟩ - ⟨τᵢⱼ⟩⟨∂Uᵢ/∂xⱼ⟩ (a sink, like the viscous dissipation)
    double_inner_product!(tke.dissipation_SGS, tke.τmean, tke.meangradU, config)
    @. tke.dissipation_SGS.values = tke.meanτgradU.values - tke.dissipation_SGS.values

    # SGS diffusion = -∂⟨u'ᵢτ'ᵢⱼ⟩/∂xⱼ, with ⟨u'ᵢτ'ᵢⱼ⟩ = ⟨Uᵢτᵢⱼ⟩ - ⟨Uᵢ⟩⟨τᵢⱼ⟩
    elementwise_multiply!(T(tke.meanUflucτfluc),T(meanU),tke.τmean,config)
    @. tke.meanUflucτfluc.x.values = tke.meanUτ.x.values - tke.meanUflucτfluc.x.values
    @. tke.meanUflucτfluc.y.values = tke.meanUτ.y.values - tke.meanUflucτfluc.y.values
    @. tke.meanUflucτfluc.z.values = tke.meanUτ.z.values - tke.meanUflucτfluc.z.values
    interpolate!(tke.Uflucτflucf,tke.meanUflucτfluc,config)
    fluctuation_boundaries!(tke.Uflucτflucf,tke.meanUflucτfluc,UBCs,time,config)
    div!(tke.diffusion_SGS,tke.Uflucτflucf,config)
    @. tke.diffusion_SGS.values = -tke.diffusion_SGS.values

    ## The convection term = -⟨Uⱼ⟩∂k/∂xⱼ ##
    inner_product!(tke.convection,meanU,tke.gradk.result,config)
    @. tke.convection.values = -tke.convection.values
    return nothing
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
