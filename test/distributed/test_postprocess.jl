# Runtime post-processing on a distributed mesh: FieldAverage, FieldRMS, ReynoldsStress, Qcriterion
# and Probe match a serial run (piso!, simple!, laplace!), probe files are written once by rank 0,
# and post-processed fields are written to the processor directories.
using XCALibre, PETSc, MPI, Test, StaticArrays

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

include(joinpath(@__DIR__, "psimple_case.jl"))
include(joinpath(@__DIR__, "laplace_case.jl"))

tmp = MPI.bcast(rank == 0 ? mktempdir() : nothing, comm; root=0)
rank == 0 && foreach(d -> mkpath(joinpath(tmp, d)), ("serial", "dist"))
MPI.Barrier(comm)

with_config(config, postprocess) = Configuration(solvers=config.solvers, schemes=config.schemes,
    runtime=config.runtime, hardware=config.hardware, boundaries=config.boundaries,
    postprocess=postprocess)

relerr(a, b) = maximum(abs.(a .- b); init=0.0) / max(maximum(abs.(b); init=0.0), eps())
read_probe(file) = [parse.(Float64, split(l, ',')) for l ∈ readlines(file)[2:end]]

# runs `case` serially on rank 0 and distributed on all ranks; returns, on rank 0, the serial and
# gathered post-processed fields (by name) and the probe files of both runs
function compare(case, gmesh_fn, names, probes; run_kw=(;))
    ser = nothing
    if rank == 0
        cd(joinpath(tmp, "serial")) do
            model, config, pp, fields = case(gmesh_fn())
            run!(model, with_config(config, pp); progress=false, run_kw...)
            ser = Dict(n => collect(f.values) for (n, f) ∈ zip(names, fields(pp)))
        end
    end
    gmesh = rank == 0 ? gmesh_fn() : nothing
    dm = distribute(gmesh; comm)
    dis = Dict{String,Any}()
    cd(joinpath(tmp, "dist")) do
        model, config, pp, fields = case(dm)
        run!(model, with_config(config, pp); progress=false, run_kw...)
        for (n, f) ∈ zip(names, fields(pp))
            dis[n] = gather(f, dm)
        end
    end
    MPI.Barrier(comm)
    files = [(read_probe(joinpath(tmp, "serial", p * ".csv")), read_probe(joinpath(tmp, "dist", p * ".csv")))
             for p ∈ probes]
    dm, ser, dis, files
end

# transient piso!: every accumulator type, two probes
nsteps = 20
function piso_case(mesh)
    model, config = incompressible_case(mesh, cavity_bcs; iterations=nsteps, time=Transient(),
        time_step=0.005, rtol=1e-12)
    (; U, p) = model.momentum
    rs = ReynoldsStress(U)
    pp = [FieldAverage(U; name="U_mean"), FieldRMS(U; name="U_rms"),
          FieldAverage(p; name="p_mean"), FieldRMS(p; name="p_rms"), rs, Qcriterion(U),
          Probe(U, mesh; location=[0.021, 0.013, 0.0], name="probe_U"),
          Probe(p, mesh; location=[0.005, 0.034, 0.0], name="probe_p"),
          FieldModes(U; name="U_modes", frequency=20, modes=1:2)] # two periods of 10 steps
    fields(pp) = (pp[1].mean.x, pp[1].mean.y, pp[2].rms.x, pp[2].rms.y, pp[3].mean, pp[4].rms,
                  pp[5].rs.xx, pp[5].rs.xy, pp[5].rs.yy, pp[6].Q,
                  pp[9].mean.x, pp[9].re[1].x, pp[9].im[1].y, pp[9].re[2].y, pp[9].im[2].x)
    model, config, pp, fields
end
piso_names = ["U_mean.x", "U_mean.y", "U_rms.x", "U_rms.y", "p_mean", "p_rms",
              "R.xx", "R.xy", "R.yy", "Q",
              "modes.mean.x", "modes.re1.x", "modes.im1.y", "modes.re2.y", "modes.im2.x"]
dm, ser, dis, files = compare(piso_case, cavity_mesh, piso_names, ["probe_U", "probe_p"])

if rank == 0
    @testset "piso! post-processing vs serial" begin
        for n ∈ piso_names
            e = relerr(dis[n], ser[n])
            println("piso $n: $e")
            @test maximum(abs, ser[n]) > 0 # non-vacuous
            @test e < 1e-9
        end
        for (s, d) ∈ files
            @test length(d) == length(s) == nsteps # one row per step, written once
            @test first.(d) == first.(s)
            @test maximum(maximum(abs.(d[i][2:end] .- s[i][2:end])) for i ∈ eachindex(s)) < 1e-9
        end
    end
end

# steady simple!
function simple_case(mesh)
    model, config = incompressible_case(mesh, bfs_bcs; iterations=10, rtol=1e-12)
    (; U, p) = model.momentum
    pp = [FieldAverage(U; name="U_mean", start=3),
          Probe(p, mesh; location=[0.1, 0.01, 0.0], name="probe_simple")]
    model, config, pp, pp -> (pp[1].mean.x, pp[1].mean.y)
end
dm, ser, dis, files = compare(simple_case, bfs_mesh, ["U_mean.x", "U_mean.y"], ["probe_simple"])
if rank == 0
    @testset "simple! post-processing vs serial" begin
        for n ∈ ("U_mean.x", "U_mean.y")
            @test relerr(dis[n], ser[n]) < 1e-9
        end
        s, d = only(files)
        @test length(d) == length(s) == 10
        @test maximum(abs(d[i][2] - s[i][2]) for i ∈ eachindex(s)) < 1e-9
    end
end

# laplace! with decomposed OpenFOAM output of the post-processed field
liters = 5
function laplace_pp_case(mesh)
    model, config = laplace_case(mesh, box_bcs; iterations=liters, convergence=1e-20)
    config = Configuration(solvers=config.solvers, schemes=config.schemes,
        runtime=Runtime(iterations=liters, write_interval=liters, time_step=1),
        hardware=config.hardware, boundaries=config.boundaries)
    (; T) = model.energy
    pp = [FieldAverage(T; name="T_mean"), Probe(T, mesh; location=[0.3, 0.6, 0.45], name="probe_T")]
    model, config, pp, pp -> (pp[1].mean,)
end
dm, ser, dis, files = compare(laplace_pp_case, box_mesh, ["T_mean"], ["probe_T"];
    run_kw=(; output=OpenFOAM()))
if rank == 0
    @testset "laplace! post-processing vs serial" begin
        @test relerr(dis["T_mean"], ser["T_mean"]) < 1e-9
        s, d = only(files)
        @test length(d) == length(s) == liters
        @test maximum(abs(d[i][2] - s[i][2]) for i ∈ eachindex(s)) < 1e-9
    end
end
@testset "post-processed field in the processor directory (rank $rank)" begin
    f = joinpath(tmp, "dist", "processor$rank", "$liters", "T_mean")
    @test isfile(f)
    vals = XCALibre.Distribute._read_foam_lists(f, Float64)[1]
    @test length(vals) == dm.partition.n_owned
end

# FieldModes split into spanwise wavenumbers: the spanwise lines of cells cross the partition. A field
# of known waves is stepped by hand on every rank.
span_mesh = rank == 0 ? UNV3D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "box32x32x32.unv")) : nothing
span = MPI.bcast(rank == 0 ? (z = [c.centre[3] for c ∈ span_mesh.cells]; (maximum(z) - minimum(z))*32/31) : nothing, comm; root=0)
dm3 = distribute(span_mesh; comm)
let
    model = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=1e-3), turbulence=RANS{Laminar}(),
        energy=Energy{Isothermal}(), domain=dm3)
    p = model.momentum.p
    x, y, z = ([dm3.cells[i].centre[k] for i ∈ eachindex(p.values)] for k ∈ 1:3) # ghosts included
    β, frequency = 2π/span, 2.0
    ω = 2π*frequency
    profile = @. (1 + x)*(2 + y^2)
    waves = Dict((2, 0) => (0.5, 0.3), (1, 1) => (0.2, 1.1), (1, -1) => (0.1, -0.4))
    u(t) = profile .* (1 .+ sum(A .* cos.(n*β .* z .- m*ω*t .- φ) for ((m, n), (A, φ)) ∈ waves))
    iterations, dt = 32, 1/(frequency*32)
    fm = FieldModes(p; name="p_modes", frequency=frequency, modes=1:2)
    stepping = convert_time_to_iterations(fm, model, dt, iterations)
    for iter ∈ 1:iterations
        p.values .= u(iter*dt)
        runtime_postprocessing!(stepping, iter, iterations, nothing, iter*dt, nothing)
    end
    sm = spanwise_modes(fm, dm3; wavenumber=β, modes=[(0, 0); collect(keys(waves)); (2, 1)])
    owned = 1:dm3.partition.n_owned
    @testset "spanwise modes across the partition (rank $rank)" begin
        for ((m, n), (A, φ)) ∈ waves
            j = findfirst(==((m, n)), sm.modes)
            @test sm.re[j].values[owned] ≈ (profile .* (A/2*cos(φ)))[owned] rtol = 1e-10
            @test sm.im[j].values[owned] ≈ (profile .* (-A/2*sin(φ)))[owned] rtol = 1e-10
        end
        @test maximum(abs, mode_amplitude(sm, (2, 1)).values[owned]; init=0.0) < 1e-10*maximum(profile)
        @test reconstruct_modes(sm, 0.123).values ≈ u(0.123) rtol = 1e-10 # ghost cells too
    end
end

# rank 0 removes its temporary directory at exit: wait until every rank has read its files
MPI.Barrier(comm)
