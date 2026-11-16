using XCALibre
using LinearAlgebra
# TKEBudget against manufactured statistics. Two samples U = ⟨U⟩ ± u' with a uniform fluctuation
# u' = (c, d, 0), a velocity gradient ∂U/∂y = γ ± g, a pressure p = ± q x and a uniform eddy
# viscosity give, away from the boundaries,
#   production          = -c d γ
#   dissipation         = -ν g²
#   dissipation_SGS     = -νₜ g²     (τ' = -νₜ(∇u' + ∇u'ᵀ) and ∇u' has one off-diagonal entry)
#   diffusion_pressure  = -q c       (⟨p'u'⟩ = q x (c, d, 0))
# and zero convection, turbulent, viscous and SGS diffusion (k and ⟨u'τ'⟩ are uniform and the
# third moment of a symmetric fluctuation vanishes).

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV2D_mesh(joinpath(grids_dir, "quad40.unv"), scale=0.001)
backend = CPU()
hardware = Hardware(backend=backend, workgroup=1024)
mesh_dev = adapt(backend, mesh)

nu, nut = 1e-3, 5e-3
c, d, γ, g, q = 0.3, 0.2, 2.0, 0.5, 1.5

function tke_case(turbulence, UBCs)
    model = Physics(
        time = Transient(),
        fluid = Fluid{Incompressible}(nu=nu),
        turbulence = turbulence,
        energy = Energy{Isothermal}(),
        domain = mesh_dev)
    BCs = assign(region=mesh_dev, (
        U = UBCs,
        p = [Zerogradient(:inlet), Zerogradient(:outlet), Zerogradient(:top), Zerogradient(:bottom)]))
    U = model.momentum.U
    gradU = Grad{Gauss}(U)
    S = StrainRate(gradU, XCALibre.T(gradU), U, FaceVectorField(mesh_dev))
    config = (hardware=hardware, runtime=(write_interval=-1,), boundaries=BCs)
    model, S, config
end

function sample!(tke, model, S, config, iterations; samples=iterations)
    (; U, p) = model.momentum
    x = [cell.centre[1] for cell ∈ mesh.cells]
    y = [cell.centre[2] for cell ∈ mesh.cells]
    for iter ∈ 1:samples
        s = isodd(iter) ? 1.0 : -1.0
        @. U.x.values = γ*y + s*c
        @. U.y.values = s*d
        @. p.values = s*q*x
        S.gradU.result.xy.values .= γ + s*g
        runtime_postprocessing!(tke, iter, iterations, S, iter*0.1, config)
    end
end

interior = setdiff(1:length(mesh.cells), mesh.boundary_cellsID)
zerogradient = [Zerogradient(:inlet), Zerogradient(:outlet), Zerogradient(:top), Zerogradient(:bottom)]

@testset "TKEBudget manufactured statistics" begin
    model, S, config = tke_case(LES{Smagorinsky}(), zerogradient)
    model.turbulence.nut.values .= nut
    iterations = 4
    tke = convert_time_to_iterations(TKEBudget(model), model, 0.1, iterations)
    @test (tke.start, tke.stop, tke.update_interval) == (1, iterations, 1)

    # terms are evaluated on write iterations and on the last one only
    sample!(tke, model, S, config, iterations; samples=iterations - 1)
    @test all(iszero, tke.production.values)
    tke = convert_time_to_iterations(TKEBudget(model), model, 0.1, iterations)
    sample!(tke, model, S, config, iterations)

    @test tke.production.values ≈ fill(-c*d*γ, length(mesh.cells))
    @test tke.dissipation.values ≈ fill(-nu*g^2, length(mesh.cells))
    @test tke.dissipation_SGS.values ≈ fill(-nut*g^2, length(mesh.cells))
    @test tke.k.values ≈ fill((c^2 + d^2)/2, length(mesh.cells))
    @test tke.diffusion_pressure.values[interior] ≈ fill(-q*c, length(interior))
    tol = 1e-10
    @test maximum(abs, tke.diffusion_turbulent.values) < tol
    @test maximum(abs, tke.diffusion_SGS.values) < tol
    @test maximum(abs, tke.diffusion_viscous.values) < tol
    @test maximum(abs, tke.convection.values) < tol

    # one output field per term, also from a vector of post-processing objects
    args = XCALibre.IOFormats.build_args([FieldAverage(model.momentum.U; name="U_mean"), tke])
    @test first.(args) == ("U_mean", tke.names...)
    @test last.(args)[2:end] == (tke.convection, tke.production, tke.diffusion_pressure,
        tke.diffusion_turbulent, tke.diffusion_viscous, tke.diffusion_SGS, tke.dissipation,
        tke.dissipation_SGS)
end

@testset "TKEBudget laminar model, walls and averaging window" begin
    walls = [Wall(:inlet, [0.0, 0.0, 0.0]), Wall(:outlet, [0.0, 0.0, 0.0]),
        Wall(:top, [0.0, 0.0, 0.0]), Wall(:bottom, [0.0, 0.0, 0.0])]
    model, S, config = tke_case(RANS{Laminar}(), walls)
    iterations = 6
    # window: iterations 3 to 6 (an even number of samples, so the mean is exact)
    tke = convert_time_to_iterations(TKEBudget(model; start=0.3), model, 0.1, iterations)
    @test (tke.start, tke.stop, tke.update_interval) == (3, iterations, 1)
    sample!(tke, model, S, config, iterations)

    @test tke.production.values ≈ fill(-c*d*γ, length(mesh.cells))
    @test tke.dissipation.values ≈ fill(-nu*g^2, length(mesh.cells))
    @test all(iszero, tke.dissipation_SGS.values) # no modelled stress
    @test all(iszero, tke.diffusion_SGS.values)

    # k = 0 on the walls: viscous diffusion removes k from the wall cells only
    wall_cells = unique(mesh.boundary_cellsID)
    @test all(<(0), tke.diffusion_viscous.values[wall_cells])
    # fluctuations vanish on the walls, so the pressure flux through them is zero and the
    # pressure diffusion integrates to zero over the domain
    volumes = [cell.volume for cell ∈ mesh.cells]
    @test abs(dot(tke.diffusion_pressure.values, volumes)) < 1e-10*sum(volumes)
    @test all(isfinite, tke.convection.values)

    @test_throws ArgumentError TKEBudget(model; names=["a", "b"])
end
