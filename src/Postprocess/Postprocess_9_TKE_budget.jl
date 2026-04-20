export TKEBudget

@kwdef struct TKEBudget{T<:AbstractField,S<:AbstractString,T1<:AbstractField,T2<:AbstractField,T3<:AbstractField,T4<:AbstractField, T5<:AbstractField,T6<:AbstractField}
    
    names
    meanU 
    meanUU 
    rst
    meangradU

    meanp 
    meanpU
    meanpUfluc
    meanpUflucf
    meanUiUiUj
    meanuiuiuj
    meanuiuiujf
    gradU2 
    meangradU2


    production
    diffusion_pressure
    diffusion_turbulent
    diffusion_viscous
    dissipation
    start::Union{Real,Nothing}
    stop::Union{Real,Nothing}
    update_interval::Union{Real,Nothing}
end



function runtime_postprocessing!(tke::TKEBudget,iter::Integer,n_iterations::Integer,config,Str,model,time)
    if must_calculate(D,iter,n_iterations)
        n = div(iter - tke.start,tke.update_interval) + 1
        U = model.momentum.U
        p = model.momentum.p
        gradU = Str.gradU.result

        ###### The production term ######

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

        ###### The Dissipation term ######

        #I need to calculate the mean of gradU squared and the mean of gradU 
        magnitude2!(tke.gradU2, gradU, config) #current value of gradU squared store in DR.GradU2

        #update running mean of gradU squared
        _update_running_mean!(tke.meangradU2, tke.gradU2,n)

        #now calculate the dissipation rate and store 
        magnitude2!(tke.dissipation, tke.meangradU, config; scale_factor = -1.0) # this calculates -1 * the magnitude of of time averaged gradU

        #finally ϵ is the sum of mean(gradU²) - mean(gradU)²
        @. tke.dissipation.values += tke.meangradU2.values


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

        ## Diffusion due to viscosity ##
        


    end

    return nothing
end