#testing
using XCALibre
grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "trig40.unv"
mesh_file = joinpath(grids_dir, grid)
mesh = UNV2D_mesh(mesh_file, scale=0.001)

mesh_dev = mesh

meshData = XCALibre.VTK.initialise_writer(mesh)

#T = ScalarField(mesh)

#write_vtk("test", mesh, meshData, ("T", T))

#Define my scalar field ϕ
#= make a new solver for my equation
div(rho*U*ϕ)=div(Γgrad(\phi)+Source

    =#
#then solve 

ϕ = ScalarField(mesh)
#function setup_convection_diffusionsolver(model)

    @info "Extracting configuration and input fields..."
    
    #(; U, p) = model.momentum
    ϕ = model.momentum
    mesh = model.domain

    @info "Pre-allocating fields..."

    #∇ϕ = Grad{schemes.ϕ.gradient}(ϕ)


    "∇p = Grad{schemes.p.gradient}(p)
    mdotf = FaceScalarField(mesh)
    rDf = FaceScalarField(mesh)
    nueff = FaceScalarField(mesh)
    # initialise!(rDf, 1.0)
    rDf.values .= 1.0
    divHv = ScalarField(mesh)
    "
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

#end