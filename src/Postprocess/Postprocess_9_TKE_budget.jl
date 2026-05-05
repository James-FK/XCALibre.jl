export TKEBudget

@kwdef struct TKEBudget{V<:AbstractVector, T1<:AbstractVectorField, T2<:AbstractField, T3<:AbstractScalarField,
    T4<:AbstractField, T5<:AbstractField, T6<:AbstractField, T7<:AbstractField, T8<:AbstractField}
    names::V
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
    gradU2::T3
    meangradU2::T6
    k::T3
    kf::T7
    gradk::Grad{<:Any}
    gradkf::T8
    production::T3
    diffusion_pressure::T3
    diffusion_turbulent::T3
    diffusion_viscous::T3
    dissipation::T3
    start::Union{Real,Nothing}
    stop::Union{Real,Nothing}
    update_interval::Union{Real,Nothing}
end
function TKEBudget(field;
    names::Vector = ["production", "diffusion_pressure", "diffusion_turbulent","diffusion_viscous", "dissipation"],
    start::Union{Real,Nothing}=nothing,
    stop::Union{Real,Nothing}=nothing,
    update_interval::Union{Real,Nothing}=nothing)


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
        gradU2 = ScalarField(field.mesh)
        meangradU2 = ScalarField(field.mesh)
        k = ScalarField(field.mesh)
        kf = FaceScalarField(field.mesh)
        gradk = Grad{Gauss}(ScalarField(field.mesh))
        gradkf = FaceVectorField(field.mesh)
        production = ScalarField(field.mesh)
        diffusion_pressure = ScalarField(field.mesh)
        diffusion_turbulent = ScalarField(field.mesh)
        diffusion_viscous = ScalarField(field.mesh)
        dissipation = ScalarField(field.mesh)

    else
        throw(ArgumentError("Unsupported field type: $(typeof(field))"))
    end
        return TKEBudget(
        names = names,
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
        gradU2 = gradU2,
        meangradU2 = meangradU2,
        k = k,
        kf = kf,
        gradk = gradk,
        gradkf = gradkf,
        production = production,
        diffusion_pressure = diffusion_pressure,
        diffusion_turbulent = diffusion_turbulent,
        diffusion_viscous = diffusion_viscous,
        dissipation = dissipation,
        start = start,
        stop = stop,
        update_interval = update_interval
    )
end


function runtime_postprocessing!(tke::TKEBudget,iter::Integer,n_iterations::Integer,config,Str,model,time)
    if must_calculate(tke,iter,n_iterations)
        n = div(iter - tke.start,tke.update_interval) + 1
        U = model.momentum.U
        p = model.momentum.p
        gradU = Str.gradU.result

        ###### The production term = − ⟨u'ᵢu'ⱼ⟩⟨∂u'ᵢ/∂xⱼ⟩  ###### 

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

        # this term is calculated using the reynolds decomposition ⟨∂u'ᵢ/∂xⱼ ∂u'ᵢ/∂xⱼ⟩ = ⟨∂Uᵢ/∂xⱼ ∂Uᵢ/∂xⱼ⟩ − ⟨∂Uᵢ/∂xⱼ⟩ ⟨∂Uᵢ/∂xⱼ⟩

        #I need to calculate the mean of gradU squared and the mean of gradU 
        magnitude2!(tke.gradU2, gradU, config) #current value of gradU squared store in DR.GradU2

        #update running mean of gradU squared
        _update_running_mean!(tke.meangradU2, tke.gradU2,n)

        #now calculate the dissipation rate and store 
        magnitude2!(tke.dissipation, tke.meangradU, config; scale_factor = -1.0) # this calculates -1 * the magnitude of of time averaged gradU

        #⟨∂u'ᵢ/∂xⱼ ∂u'ᵢ/∂xⱼ⟩ is the sum of mean(gradU²) - mean(gradU)²
        @. tke.dissipation.values += tke.meangradU2.values
        #finally scale by -ν to get the dissipation term
        @. tke.dissipation.values *= (-1 * model.fluid.nu.values)


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
        div!(tke.diffusion_pressure,tke.meanpUflucf,config)
        #finally scale by the density 
        @. tke.diffusion_pressure.values = -tke.diffusion_pressure.values / model.fluid.rho.values
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
        div!(tke.diffusion_turbulent,tke.meanuiuiujf,config)
        #finally scale by -1/2
        @. tke.diffusion_turbulent.values *= -0.5
        ## Diffusion due to viscosity ##

        
        # get k from the 1/2 the trace of the reynolds stress tensor
        @. tke.k.values = 0.5 * (tke.rst.xx.values + tke.rst.yy.values + tke.rst.zz.values )
        #now just need the laplacian of k 
        # gradk = Grad{Gauss}(tke.k) # this needs to be done outside the loop
        interpolate!(tke.kf,tke.k,config)
        green_gauss!(tke.gradk,tke.kf,config) #calculate gradk

        #finally just calculate divergence of grad k 
        interpolate!(tke.gradkf,tke.gradk.result,config)
        div!(tke.diffusion_viscous,tke.gradkf,config)
        @. tke.diffusion_viscous.values = tke.diffusion_viscous.values * model.fluid.nu.values

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
        gradU2 = tke.gradU2,
        meangradU2 = tke.meangradU2,
        k = tke.k,
        kf = tke.kf,
        gradk = tke.gradk,
        gradkf = tke.gradkf,
        production = tke.production,
        diffusion_pressure = tke.diffusion_pressure,
        diffusion_turbulent = tke.diffusion_turbulent,
        diffusion_viscous = tke.diffusion_viscous,
        dissipation = tke.dissipation,
        start = start,
        stop = stop,
        update_interval = update_interval
    )
end