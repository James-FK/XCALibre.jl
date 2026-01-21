export WALE

# Model type definition
"""
    WALE <: AbstractTurbulenceModel

WALE LES model containing all WALE field parameters.

### Fields
- `nut` -- Eddy viscosity ScalarField.
- `nutf` -- Eddy viscosity FaceScalarField.
- `coeffs` -- Model coefficients.

"""
struct WALE{S1,S2,C} <: AbstractLESModel
    nut::S1
    nutf::S2
    coeffs::C
end
Adapt.@adapt_structure WALE

struct WALEModel{T,D,S1}
    turbulence::T
    Δ::D 
    state::S1
end
Adapt.@adapt_structure WALEModel

# Model API constructor (pass user input as keyword arguments and process as needed)
LES{WALE}(; C=0.325) = begin 
    coeffs = (C=C,)
    ARG = typeof(coeffs)
    LES{WALE,ARG}(coeffs)
end

# Functor as constructor (internally called by Physics API): Returns fields and user data
(les::LES{WALE, ARG})(mesh) where ARG = begin
    nut = ScalarField(mesh)
    nutf = FaceScalarField(mesh)
    coeffs = les.args
    WALE(nut, nutf, coeffs)
end

# Model initialisation
"""
    initialise(turbulence::WALE, model::Physics{T,F,SO,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,SO,M,Tu,E,D,BI}

Initialisation of turbulent transport equations.

### Input
- `turbulence`: turbulence model.
- `model`: Physics model defined by user.
- `mdtof`: Face mass flow.
- `peqn`: Pressure equation.
- `config`: Configuration structure defined by user with solvers, schemes, runtime and hardware structures set.

### Output
Returns a structure holding the fields and data needed for this model

    WALEModel(
        turbulence, 
        Δ, 
        ModelState((), false)
    )

"""
function initialise(
    turbulence::WALE, model::Physics{T,F,SO,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,SO,M,Tu,E,D,BI}

    (; solvers, schemes, runtime, boundaries) = config
    mesh = model.domain
    
    Δ = ScalarField(mesh)

    delta!(Δ, mesh, config)
    (; coeffs) = model.turbulence
    @. Δ.values = (Δ.values*coeffs.C)^2.0
    
    return WALEModel(
        turbulence, 
        Δ, 
        ModelState((), false)
    ), config
end

# Model solver call (implementation)
"""
    turbulence!(les::WALEModel, model::Physics{T,F,SO,M,Tu,E,D,BI}, S, S2, prev, time, config
    ) where {T,F,SO,M,Tu<:AbstractTurbulenceModel,E,D,BI}

Run turbulence model transport equations.

### Input
- `les::WALEModel`: `WALE` LES turbulence model.
- `model`: Physics model defined by user.
- `S`: Strain rate tensor.
- `S2`: Square of the strain rate magnitude.
- `prev`: Previous field.
- `time`: current simulation time 
- `config`: Configuration structure defined by user with solvers, schemes, runtime and hardware structures set.

"""
function turbulence!(
    les::WALEModel, model::Physics{T,F,SO,M,Tu,E,D,BI}, S, prev, time, config
    ) where {T,F,SO,M,Tu<:AbstractTurbulenceModel,E,D,BI}

    mesh = model.domain
    scalar = ScalarFloat(mesh)
    squaredgradU = TensorField(mesh)
    symmetricUSquare = TensorField(mesh)
    squaredgradU = TensorField(mesh)
    (; boundaries, hardware) = config
    (; backend, workgroup) = hardware
    (; nut, nutf, coeffs) = les.turbulence
    (; U, Uf, gradU) = S
    (; Δ) = les

    
    grad!(gradU, Uf, U, boundaries.U, time, config) # update gradient (and S)
    limit_gradient!(config.schemes.U.limiter, gradU, U, config)
    
    wk = _setup(backend, workgroup, length(nut))[2] # index 2 to extract the workgroup
    AK.foreachindex(nut, min_elems=wk, block_size=wk) do i
        Si = S[i] # 0.5*(gradUi + gradUi')
        magS = (Si⋅Si)
        
        #first need square of the velocity gradient tensor 
        squaredgradU[i] = gradU[i] * gradU[i]

        #now i need the symmetric part of this tensor 
        sq = squaredgradU[i]
        symmetricUSquare[i] = 0.5 * (sq + sq')
        #then compute the deviatoric 
        devB = Dev(symmetricUSquare)
        Sd = devB[i]
        #finally need a scalarfield based on the magnitudes 
        magSd = Sd⋅Sd
        mag = magSd^3/2/(magS^5/4+magSd^5/4)

        nut[i] = Δ[i]*mag # Δ is (Cw*Δ)^2
    end

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, boundaries.nut, time, config)
    correct_eddy_viscosity!(nutf, boundaries.nut, model, config)
end

# Specialise VTK writer
function save_output(model::Physics{T,F,SO,M,Tu,E,D,BI}, outputWriter, iteration, time, config
    ) where {T,F,SO,M,Tu<:WALE,E,D,BI}
    args = (
        ("U", model.momentum.U), 
        ("p", model.momentum.p),
        ("nut", model.turbulence.nut)
    )
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end