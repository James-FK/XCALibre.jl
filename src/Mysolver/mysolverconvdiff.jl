#testing
using XCALibre
using Accessors
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "quad40.unv"
mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

backend = CPU()
hardware = set_hardware(backend=backend, workgroup=4)


#meshData = XCALibre.VTK.initialise_writer(mesh)



ϕ = ScalarField(mesh)
rhoU = FaceScalarField(mesh)


divrhoU = FaceScalarField(mesh)
Γ= FaceScalarField(mesh)
ϕ_source = ScalarField(mesh)


ϕ_eqn = (
Divergence{Upwind}(divrhoU, ϕ) 
-Laplacian{Linear}(Γ, ϕ) 
== 
Source(ϕ_source)
) → ScalarEquation(mesh)  


velocity = 1.0


ϕ = assign(ϕ,    Dirichlet(:inlet, velocity),
Dirichlet(:outlet, 0.0),
Neumann(:bottom, 0.0),
Neumann(:top, 0.0))
#outlet should be diriclet

schemes = (; ϕ = set_schemes(divergence = Linear))
Γ.values .= 0.1 
rhoU.values .= 0.0


solvers = (
    ϕ = set_solver(
        ϕ;
        solver = BicgstabSolver,
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax = 0.7,
        rtol = 1e-4,
        atol = 1e-14,
        itmax = 500
    ),
    )
    
runtime = set_runtime(iterations=200, time_step=1, write_interval=200)
config = Configuration(solvers = solvers, schemes = schemes, runtime = runtime, hardware= hardware)

@reset ϕ_eqn.preconditioner = set_preconditioner(solvers.ϕ.preconditioner, ϕ_eqn, ϕ.BCs, config)
@reset ϕ_eqn.solver = solvers.ϕ.solver(_A(ϕ_eqn), _b(ϕ_eqn))

initialise!(ϕ, 1.0)


solve_equation!(ϕ_eqn, ϕ, solvers.ϕ, config)
#currently my e

meshData = initialise_writer(mesh)
write_vtk("test", mesh, meshData, ("phi", ϕ))