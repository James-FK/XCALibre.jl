#the production of TKE is the inner product between the reynolds stress tensor and the time averaged velocity gradient tensor ∇u
export ProductionTKE
@kwdef struct ProductionTKE{T<:AbstractField,T1<:AbstractField,T2<:AbstractField,T3<:AbstractField,T4<:AbstractField, S<:AbstractString}
    field::T 
    name::S
    mean::T1
    mean_sq::T2
    RST::T2
    gradU_mean::T3
    result::T4
    start::Union{Real,Nothing}
    stop::Union{Real,Nothing}
    update_interval::Union{Real,Nothing}
end  

function ProductionTKE(field; name::String =  "Production_TKE", start::Union{Real,Nothing}=nothing, stop::Union{Real,Nothing}=nothing,update_interval::Union{Real,Nothing}=nothing)
    if field isa VectorField
        RST = SymmetricTensorField(field.mesh)
        mean = VectorField(field.mesh)
        mean_sq = SymmetricTensorField(field.mesh)
        result = ScalarField(field.mesh)
        gradU_mean = TensorField(field.mesh)
    else
        throw(ArgumentError("Unsupported field type: $(typeof(field))"))
    end
    return  ProductionTKE(field=field, name=name, RST=RST, mean=mean, mean_sq=mean_sq,result=result,gradU_mean=gradU_mean, start=start, stop=stop, update_interval=update_interval)
end

function runtime_postprocessing!(P::ProductionTKE{T,T1,T2,T3,T4,S},iter::Integer,n_iterations::Integer,config,Str,model,time) where {T<:VectorField,T1,T2,T3,T4,S}
    if must_calculate(P,iter,n_iterations)
        n = div(iter - P.start,P.update_interval) + 1
        current_field = P.field
        gradU = Str.gradU.result
        _update_running_mean!(P.mean.x.values, current_field.x.values, n)
        _update_running_mean!(P.mean.y.values, current_field.y.values, n)
        _update_running_mean!(P.mean.z.values, current_field.z.values, n)
    

        _update_running_mean!(P.mean_sq.xx.values, current_field.x.values .^2,n)
        _update_running_mean!(P.mean_sq.xy.values, current_field.x.values .* current_field.y.values,n)
        _update_running_mean!(P.mean_sq.xz.values, current_field.x.values .* current_field.z.values,n)
        _update_running_mean!(P.mean_sq.yy.values, current_field.y.values .^2,n)
        _update_running_mean!(P.mean_sq.yz.values, current_field.y.values .* current_field.z.values,n)
        _update_running_mean!(P.mean_sq.zz.values, current_field.z.values .^2,n)

        #update the running mean of the velocity gradient tensor 
        _update_running_mean!(P.gradU_mean.xx.values, gradU.xx.values,n)
        _update_running_mean!(P.gradU_mean.xy.values, gradU.xy.values,n)
        _update_running_mean!(P.gradU_mean.xz.values, gradU.xz.values,n)
        _update_running_mean!(P.gradU_mean.yx.values, gradU.yx.values,n)
        _update_running_mean!(P.gradU_mean.yy.values, gradU.yy.values,n)
        _update_running_mean!(P.gradU_mean.yz.values, gradU.yz.values,n)
        _update_running_mean!(P.gradU_mean.zx.values, gradU.zx.values,n)
        _update_running_mean!(P.gradU_mean.zy.values, gradU.zy.values,n)
        _update_running_mean!(P.gradU_mean.zz.values, gradU.zz.values,n)


        #calculate and store reynolds stress tensor 
        @. P.RST.xx.values = P.mean_sq.xx.values - P.mean.x.values^2
        @. P.RST.xy.values = P.mean_sq.xy.values - P.mean.x.values * P.mean.y.values
        @. P.RST.xz.values = P.mean_sq.xz.values - P.mean.x.values * P.mean.z.values

        @. P.RST.yy.values = P.mean_sq.yy.values - P.mean.y.values^2
        @. P.RST.yz.values = P.mean_sq.yz.values - P.mean.y.values * P.mean.z.values

        @. P.RST.zz.values = P.mean_sq.zz.values - P.mean.z.values^2

        #finally evaluate the double inner product (double contraction) of the Reynolds stress tensor with the time averaged velocity gradient tensor
        double_inner_product!(P.result, P.RST, P.gradU_mean,config; scale_factor = -1.0)

    end
    return nothing
end
function convert_time_to_iterations(P::ProductionTKE, model,dt,iterations)
    if model.time === Transient()
        if P.start === nothing
            start = 1
        else 
            P.start >= 0  || throw(ArgumentError("Start must be a value ≥ 0 (got $(P.start))"))
            start = clamp(ceil(Int, P.start / dt), 1, iterations) 
        end

        if P.stop === nothing 
            stop = iterations
        else
            P.stop ≥ 0 || throw(ArgumentError("stop must be ≥ 0 (got $(P.stop))"))
            stop = clamp(floor(Int,P.stop / dt), 1, iterations)
        end

        if P.update_interval === nothing 
            update_interval = 1
        else
            P.update_interval > 0 || throw(ArgumentError("update interval must be > 0 (got $(P.update_interval))"))
            update_interval = max(1, floor(Int,P.update_interval / dt))
        end
        stop >= start || throw(ArgumentError("After conversion with dt=$dt the averaging window is empty (start = $start, stop = $stop)"))
        return ProductionTKE(field=P.field, name=P.name, RST=P.RST, mean=P.mean, mean_sq=P.mean_sq,result=P.result,gradU_mean=P.gradU_mean, start=start, stop=stop, update_interval=update_interval)

    else #for Steady runs use iterations 
        if P.start === nothing
            start = 1
        else 
            P.start isa Integer || throw(ArgumentError("For steady runs, start must be specified in iterations and therefore be an integer (got $(P.start))"))
            P.start >=1     || throw(ArgumentError("Start must be ≥1 (got $(P.start))"))
            start = P.start
        end

        if P.stop === nothing 
            stop = iterations
        else
            P.stop isa Integer || throw(ArgumentError("For steady runs, stop must be specified in iterations and therefore be an integer (got $(P.stop))"))
            P.stop >=1     || throw(ArgumentError("Stop must be ≥1 (got $(P.stop))"))
            stop = P.stop
        end

        if P.update_interval === nothing 
            update_interval = 1
        else
            P.update_interval isa Integer || throw(ArgumentError("For steady runs, update_interval must be specified in iterations and therefore be an integer (got $(P.update_interval))"))
            P.update_interval >= 1 || throw(ArgumentError("update interval must be ≥1 (got $(P.update_interval))"))
            update_interval = P.update_interval
        end

        stop >= start || throw(ArgumentError("stop iteration needs to be ≥ start  (got start = $start, stop = $stop)"))
        stop <= iterations || throw(ArgumentError("stop ($stop) must be ≤ iterations ($iterations)"))
        return ProductionTKE(field=P.field, name=P.name, RST=P.RST, mean=P.mean, mean_sq=P.mean_sq,result=P.result,gradU_mean=P.gradU_mean, start=start, stop=stop, update_interval=update_interval)
    end
end
