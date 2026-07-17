export TKEBudget

@kwdef struct TKEBudget{V<:AbstractVector, T1<:AbstractVectorField, T2<:AbstractField, T3<:AbstractScalarField,
    T4<:AbstractField, T5<:AbstractField,  T7<:AbstractField, T8<:AbstractField}
    names::V
    wall_IDs_range::UnitRange{Int64}
    meanU::T1
    meanUU::T2
    rst::T2
    meangradU::T4
    meanp::T3
    meanpU::T1
    meanpUfluc::T1
    meanpUflucf::T5
    meanUiUiUj::T1
    meanuiuiuj::T1
    meanuiuiujf::T5
    gradUfluc::T4
    gradUfluc2::T3
    k::T3
    kf::T7
    gradk::Grad{<:Any}
    gradkf::T8
    τ::T2
    τmean::T2
    τfluc::T2
    τflucgradUfluc::T3
    Ufluc::T1
    Uflucτfluc::T1
    meanUflucτfluc::T1
    Uflucτflucf::T5
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
    wall_IDs_range = 0
    UBCs = BCs.U
    for b in UBCs
        if b isa Wall
            wall_IDs_range = b.IDs_range
        end
    end

    if field isa VectorField
        meanU = VectorField(field.mesh)
        meanUU = SymmetricTensorField(field.mesh)
        rst = SymmetricTensorField(field.mesh)
        meangradU = TensorField(field.mesh)
        meanp = ScalarField(field.mesh)
        meanpU = VectorField(field.mesh)
        meanpUfluc = VectorField(field.mesh)
        meanpUflucf = FaceVectorField(field.mesh)
        meanUiUiUj = VectorField(field.mesh)
        meanuiuiuj = VectorField(field.mesh)
        meanuiuiujf = FaceVectorField(field.mesh)
        gradUfluc = TensorField(field.mesh)
        gradUfluc2 = ScalarField(field.mesh)
        k = ScalarField(field.mesh)
        kf = FaceScalarField(field.mesh)
        gradk = Grad{Gauss}(ScalarField(field.mesh))
        gradkf = FaceVectorField(field.mesh)
        τ = SymmetricTensorField(field.mesh)
        τmean = SymmetricTensorField(field.mesh)
        τfluc = SymmetricTensorField(field.mesh)
        τflucgradUfluc = ScalarField(field.mesh)
        Ufluc = VectorField(field.mesh)
        Uflucτfluc = VectorField(field.mesh)
        meanUflucτfluc = VectorField(field.mesh)
        Uflucτflucf = FaceVectorField(field.mesh)
        convection = ScalarField(field.mesh)
        production = ScalarField(field.mesh)
        diffusion_pressure = ScalarField(field.mesh)
        diffusion_turbulent = ScalarField(field.mesh)
        diffusion_viscous = ScalarField(field.mesh)
        diffusion_SGS = ScalarField(field.mesh)
        dissipation = ScalarField(field.mesh)
        dissipation_SGS = ScalarField(field.mesh)

    else
        throw(ArgumentError("Unsupported field type: $(typeof(field))"))
    end
        return TKEBudget(
        names = names,
        wall_IDs_range = wall_IDs_range,
        meanU = meanU,
        meanUU = meanUU,
        rst = rst,
        meangradU = meangradU,
        meanp = meanp,
        meanpU = meanpU,
        meanpUfluc = meanpUfluc,
        meanpUflucf = meanpUflucf,
        meanUiUiUj = meanUiUiUj,
        meanuiuiuj = meanuiuiuj,
        meanuiuiujf = meanuiuiujf,
        gradUfluc = gradUfluc,
        gradUfluc2 = gradUfluc2,
        k = k,
        kf = kf,
        gradk = gradk,
        gradkf = gradkf,
        τ = τ,
        τmean = τmean,
        τfluc = τfluc,
        τflucgradUfluc = τflucgradUfluc,
        Ufluc = Ufluc,
        Uflucτfluc = Uflucτfluc, 
        meanUflucτfluc = meanUflucτfluc,
        Uflucτflucf = Uflucτflucf, 
        convection = convection,
        production = production,
        diffusion_pressure = diffusion_pressure,
        diffusion_turbulent = diffusion_turbulent,
        diffusion_viscous = diffusion_viscous,
        diffusion_SGS = diffusion_SGS,
        dissipation = dissipation,
        dissipation_SGS = dissipation_SGS,
        start = start,
        stop = stop,
        update_interval = update_interval
    )
end


function runtime_postprocessing!(tke::TKEBudget,iter::Integer,n_iterations::Integer,config,S,model,time)
    if must_calculate(tke,iter,n_iterations)
        n = div(iter - tke.start,tke.update_interval) + 1
        U = model.momentum.U
        p = model.momentum.p
        nu = model.fluid.nu.values
        nut = model.turbulence.nut
        gradU = S.gradU.result
        UBCs = config.boundaries.U
        ###### The production term = − ⟨u'ᵢu'ⱼ⟩⟨∂Uᵢ/∂xⱼ⟩  ###### 

        _update_running_mean!(tke.meanU, U, n) #update ⟨Uᵢ⟩
        _update_running_mean!(tke.meanUU,U,n) #update ⟨UᵢUⱼ⟩ 

        # store the Reynolds Stress Tensor Rᵢⱼ = ⟨UᵢUⱼ⟩ - ⟨Uᵢ⟩⟨Uⱼ⟩
        @. tke.rst.xx.values = tke.meanUU.xx.values - tke.meanU.x.values^2
        @. tke.rst.xy.values = tke.meanUU.xy.values - tke.meanU.x.values * tke.meanU.y.values
        @. tke.rst.xz.values = tke.meanUU.xz.values - tke.meanU.x.values * tke.meanU.z.values
        @. tke.rst.yy.values = tke.meanUU.yy.values - tke.meanU.y.values^2
        @. tke.rst.yz.values = tke.meanUU.yz.values - tke.meanU.y.values * tke.meanU.z.values
        @. tke.rst.zz.values = tke.meanUU.zz.values - tke.meanU.z.values^2

        #update the running mean of the velocity gradient tensor 
        _update_running_mean!(tke.meangradU, gradU,n)
        #finally evaluate the double inner product of the Reynolds stress tensor with the mean of gradU
        double_inner_product!(tke.production, tke.rst, tke.meangradU,config; scale_factor = -1.0)

        ###### The Dissipation term  = -ν⟨∂u'ᵢ/∂xⱼ ∂u'ᵢ/∂xⱼ⟩ ######

        # first get ∂u'ᵢ/∂xⱼ = ∂Uᵢ/∂xⱼ - ⟨∂Uᵢ/∂xⱼ⟩
        @. tke.gradUfluc.xx.values = gradU.xx.values - tke.meangradU.xx.values
        @. tke.gradUfluc.xy.values = gradU.xy.values - tke.meangradU.xy.values    
        @. tke.gradUfluc.xz.values = gradU.xz.values - tke.meangradU.xz.values
        @. tke.gradUfluc.yx.values = gradU.yx.values - tke.meangradU.yx.values
        @. tke.gradUfluc.yy.values = gradU.yy.values - tke.meangradU.yy.values
        @. tke.gradUfluc.yz.values = gradU.yz.values - tke.meangradU.yz.values
        @. tke.gradUfluc.zx.values = gradU.zx.values - tke.meangradU.zx.values
        @. tke.gradUfluc.zy.values = gradU.zy.values - tke.meangradU.zy.values
        @. tke.gradUfluc.zz.values = gradU.zz.values - tke.meangradU.zz.values
        #double contraction 
        magnitude2!(tke.gradUfluc2,tke.gradUfluc, config; scale_factor = (-nu))
        _update_running_mean!(tke.dissipation, tke.gradUfluc2,n)



        ###### The Diffusion terms ######  

        ## Diffusion due to pressure ## 

        _update_running_mean!(tke.meanp,p,n)
        _update_running_mean!(tke.meanpU,p * U ,n)

        meanpUflucf = tke.meanpU - (tke.meanU * tke.meanp)
        tke.meanpUfluc.x.values .= meanpUflucf.x.values
        tke.meanpUfluc.y.values .= meanpUflucf.y.values
        tke.meanpUfluc.z.values .= meanpUflucf.z.values
        #divergence of ⟨u'p'⟩
        interpolate!(tke.meanpUflucf,tke.meanpUfluc,config)
        correct_boundaries!(tke.meanpUflucf,tke.meanpUfluc,UBCs,time,config)
        div!(tke.diffusion_pressure,tke.meanpUflucf,config)
        #finally scale by -1
        @. tke.diffusion_pressure.values = -tke.diffusion_pressure.values
        ## Diffusion due to fluctuations ## 

        #update mean of ⟨UᵢUᵢUⱼ⟩
        _update_running_mean!(tke.meanUiUiUj.x.values,(U.x.values.^2 + U.y.values.^2 + U.z.values.^2) .*U.x.values,n) 
        _update_running_mean!(tke.meanUiUiUj.y.values,(U.x.values.^2 + U.y.values.^2 + U.z.values.^2) .*U.y.values,n) 
        _update_running_mean!(tke.meanUiUiUj.z.values,(U.x.values.^2 + U.y.values.^2 + U.z.values.^2) .*U.z.values,n) 


        # ⟨u'ᵢu'ᵢu'ⱼ⟩ = ⟨UᵢUᵢUⱼ⟩ - ⟨Uⱼ⟩⟨UᵢUᵢ⟩ - 2⟨Uᵢ⟩⟨UᵢUⱼ⟩ + 2⟨Uᵢ⟩⟨Uᵢ⟩⟨Uⱼ⟩
        @. tke.meanuiuiuj.x.values = (tke.meanUiUiUj.x.values - (tke.meanU.x.values * (tke.meanUU.xx.values + tke.meanUU.yy.values + tke.meanUU.zz.values))
                                        - 2 * (tke.meanU.x.values * tke.meanUU.xx.values + tke.meanU.y.values * tke.meanUU.yx.values + tke.meanU.z.values * tke.meanUU.zx.values)
                                        + 2 * (tke.meanU.x.values^2 + tke.meanU.y.values^2 + tke.meanU.z.values^2) * tke.meanU.x.values)

        @. tke.meanuiuiuj.y.values = (tke.meanUiUiUj.y.values - (tke.meanU.y.values * (tke.meanUU.xx.values + tke.meanUU.yy.values + tke.meanUU.zz.values))
                                        - 2 * (tke.meanU.x.values * tke.meanUU.xy.values + tke.meanU.y.values * tke.meanUU.yy.values + tke.meanU.z.values * tke.meanUU.zy.values)
                                        + 2 * (tke.meanU.x.values^2 + tke.meanU.y.values^2 + tke.meanU.z.values^2) * tke.meanU.y.values)

        @. tke.meanuiuiuj.z.values = (tke.meanUiUiUj.z.values - (tke.meanU.z.values * (tke.meanUU.xx.values + tke.meanUU.yy.values + tke.meanUU.zz.values))
                                        - 2 * (tke.meanU.x.values * tke.meanUU.xz.values + tke.meanU.y.values * tke.meanUU.yz.values + tke.meanU.z.values * tke.meanUU.zz.values)
                                        + 2 * (tke.meanU.x.values^2 + tke.meanU.y.values^2 + tke.meanU.z.values^2) * tke.meanU.z.values)
        interpolate!(tke.meanuiuiujf,tke.meanuiuiuj,config)
        correct_boundaries!(tke.meanuiuiujf,tke.meanuiuiuj,UBCs,time,config)
        div!(tke.diffusion_turbulent,tke.meanuiuiujf,config)
        #finally scale by -1/2
        @. tke.diffusion_turbulent.values *= -0.5

        ## Diffusion due to viscosity ##

        # get k from the 1/2 the trace of the reynolds stress tensor
        @. tke.k.values = 0.5 * (tke.rst.xx.values + tke.rst.yy.values + tke.rst.zz.values )
        #now just need the laplacian of k 
        # gradk = Grad{Gauss}(tke.k) # this needs to be done outside the loop
        interpolate!(tke.kf,tke.k,config)
        wall_correction!(tke.kf,tke.wall_IDs_range,config)
        green_gauss!(tke.gradk,tke.kf,config) #calculate gradk

        #finally just calculate divergence of grad k 
        interpolate!(tke.gradkf,tke.gradk.result,config)

        correct_gradient!(tke.gradkf, tke.k, 0.0, tke.wall_IDs_range, config)
        div!(tke.diffusion_viscous,tke.gradkf,config)

        @. tke.diffusion_viscous.values = tke.diffusion_viscous.values * model.fluid.nu.values
        


        ## Subgrid scale contributions to budget in case of LES ##

        # first compute the SGS dissipation as ϵ_SGS = -⟨ τ'ᵢⱼ ∂u'ᵢ/∂xⱼ ⟩
        elementwise_multiply!(tke.τ,nut,S,config;scale_factor = -2) #first evaluate τᵢⱼ = -2νₜ Sᵢⱼ
        _update_running_mean!(tke.τmean,tke.τ,n) # mean of τᵢⱼ is required for the fluctuation τ'ᵢⱼ

        @. tke.τfluc.xx.values = tke.τ.xx.values - tke.τmean.xx.values
        @. tke.τfluc.xy.values = tke.τ.xy.values - tke.τmean.xy.values
        @. tke.τfluc.xz.values = tke.τ.xz.values - tke.τmean.xz.values
        @. tke.τfluc.yy.values = tke.τ.yy.values - tke.τmean.yy.values
        @. tke.τfluc.yz.values = tke.τ.yz.values - tke.τmean.yz.values
        @. tke.τfluc.zz.values = tke.τ.zz.values - tke.τmean.zz.values

        double_inner_product!(tke.τflucgradUfluc,tke.τfluc,tke.gradUfluc,config; scale_factor=-1)
        _update_running_mean!(tke.dissipation_SGS,tke.τflucgradUfluc,n)


        # the contribution of SGS to diffusion is ∂/∂xⱼ⟨u'ᵢτ'ᵢⱼ⟩
        # need the full fluctuations 
        @. tke.Ufluc.x.values = U.x.values - tke.meanU.x.values
        @. tke.Ufluc.y.values = U.y.values - tke.meanU.y.values
        @. tke.Ufluc.z.values = U.z.values - tke.meanU.z.values
     
        elementwise_multiply!(T(tke.Uflucτfluc),T(tke.Ufluc),tke.τfluc,config)
        _update_running_mean!(tke.meanUflucτfluc,tke.Uflucτfluc,n)
        interpolate!(tke.Uflucτflucf,tke.meanUflucτfluc,config)
        correct_boundaries!(tke.Uflucτflucf,tke.meanUflucτfluc,UBCs,time,config)
        div!(tke.diffusion_SGS,tke.Uflucτflucf,config)

        ## The convection term ## 
        inner_product!(tke.convection,tke.meanU,tke.gradk.result,config)
    end

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

    return TKEBudget(
        names = tke.names,
        wall_IDs_range = tke.wall_IDs_range,
        meanU = tke.meanU,
        meanUU = tke.meanUU,
        rst = tke.rst,
        meangradU = tke.meangradU,
        meanp = tke.meanp,
        meanpU = tke.meanpU,
        meanpUfluc = tke.meanpUfluc,
        meanpUflucf = tke.meanpUflucf,
        meanUiUiUj = tke.meanUiUiUj,
        meanuiuiuj = tke.meanuiuiuj,
        meanuiuiujf = tke.meanuiuiujf,
        gradUfluc = tke.gradUfluc,
        gradUfluc2 = tke.gradUfluc2,
        k = tke.k,
        kf = tke.kf,
        gradk = tke.gradk,
        gradkf = tke.gradkf,
        τ = tke.τ,
        τmean = tke.τmean,
        τfluc = tke.τfluc,
        τflucgradUfluc = tke.τflucgradUfluc,
        Ufluc = tke.Ufluc,
        Uflucτfluc = tke.Uflucτfluc, 
        meanUflucτfluc = tke.meanUflucτfluc,
        Uflucτflucf = tke.Uflucτflucf, 
        convection = tke.convection,
        production = tke.production,
        diffusion_pressure = tke.diffusion_pressure,
        diffusion_turbulent = tke.diffusion_turbulent,
        diffusion_viscous = tke.diffusion_viscous,
        diffusion_SGS = tke.diffusion_SGS,
        dissipation = tke.dissipation,
        dissipation_SGS = tke.dissipation_SGS,
        start = start,
        stop = stop,
        update_interval = update_interval
    )
end