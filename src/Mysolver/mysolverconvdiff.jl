#testing
using XCALibre
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "quad40.unv"
mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)
backend = CPU()
hardware = set_hardware(backend=backend, workgroup=4)

mesh_dev = mesh

#meshData = XCALibre.VTK.initialise_writer(mesh)



ϕ = ScalarField(mesh)
rhoU = FaceScalarField(mesh)
Γ= FaceScalarField(mesh)

ϕ_source = ScalarField(mesh)


ϕ_eqn = (
Divergence{Linear}(rhoU, ϕ) 
- Laplacian{Linear}(Γ, ϕ) 
== 
Source(ϕ_source)
) → ScalarEquation(mesh)  


velocity = [1.0, 0.0, 0.0]

@assign!  ϕ (
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Wall(:wall, [0.0, 0.0, 0.0]),
    Wall(:top, [0.0, 0.0, 0.0]),
)


#schemes = (
    #ϕ = set_schemes(divergence = Linear))



solvers = (
    ϕ = set_solver(
        ϕ;
        solver = BicgstabSolver,
        preconditioner = Jacobi(),
        convergence = 1e-7,
        relax = 0.7,
        rtol = 1e-4,
        atol = 1e10
    ))


    runtime = set_runtime(iterations=200, time_step=1, write_interval=200)
config = Configuration(
    solvers = solvers, schemes = schemes, runtime = runtime, hardware= hardware
)

solve_equation!(ϕ_eqn, ϕ, solvers.ϕ, config; ref=pref)
#currently my e