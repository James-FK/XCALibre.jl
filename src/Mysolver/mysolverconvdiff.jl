#testing
using XCALibre
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "trig40.unv"
mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

mesh_dev = mesh

meshData = XCALibre.VTK.initialise_writer(mesh)



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

schemes = (
    ϕ = set_schemes(divergence = Linear))


#finally solve equation 
config = Configuration(
    solvers = solvers, schemes = schemes, runtime = runtime, hardware= hardware)








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

config = Configuration(
    solvers = solvers, schemes = schemes, runtime = runtime, hardware= hardware
)

solve_equation!(ϕ_eqn, ϕ, solvers.ϕ, config; ref=pref)
