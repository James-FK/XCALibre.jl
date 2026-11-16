using XCALibre
using LinearAlgebra

# FieldModes holds, in every cell, the Fourier coefficients in time of a field at multiples of a base
# frequency; spanwise_modes splits them into spanwise wavenumbers; reconstruct_modes gives the waves back.

# records U.x and the time at every step, for the reference Fourier sums
struct ModeRecorder{F}
    U::F
    samples::Vector{Vector{Float64}}
    times::Vector{Float64}
end
XCALibre.convert_time_to_iterations(r::ModeRecorder, model, dt, iterations) = r
function XCALibre.runtime_postprocessing!(r::ModeRecorder, iter::Integer, n::Integer, S, time, config)
    push!(r.samples, copy(r.U.x.values)); push!(r.times, time)
    nothing
end

@testset "FieldModes" begin

@testset "pulsating inlet (piso!)" begin
    mesh = UNV2D_mesh(pkgdir(XCALibre, "examples/0_GRIDS", "quad40.unv"), scale=0.001)
    backend = CPU()
    mesh_dev = adapt(backend, mesh)
    U0, A, frequency = 0.3, 0.5, 1
    velocity = [1.5*U0, U0, 0.0]

    model = Physics(
        time = Transient(),
        fluid = Fluid{Incompressible}(nu=1e-3),
        turbulence = RANS{Laminar}(),
        energy = Energy{Isothermal}(),
        domain = mesh_dev)

    @inline inflow(vec, t, i) = SVector{3}(U0*(1 + A*cospi(2*frequency*t)), U0*(1 + A*sinpi(2*frequency*t)), 0.0)
    BCs = assign(region=mesh_dev, (
        U = [DirichletFunction(:inlet, inflow), Extrapolated(:outlet),
            Symmetry(:bottom, [0.0, 0.0, 0.0]), Symmetry(:top, [0.0, 0.0, 0.0])],
        p = [Extrapolated(:inlet), Dirichlet(:outlet, 0.0), Symmetry(:bottom), Symmetry(:top)]))
    schemes = (U = Schemes(time=Euler), p = Schemes())
    solvers = (
        U = SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-7, relax=1.0, rtol=1e-3),
        p = SolverSetup(solver=Gmres(), preconditioner=Jacobi(), convergence=1e-7, relax=1.0, rtol=1e-3))

    iterations, timestep = 150, 0.01 # 1.5 periods: the window is the last whole one, steps 51 to 150
    runtime = Runtime(iterations=iterations, time_step=timestep, write_interval=iterations)
    hardware = Hardware(backend=backend, workgroup=1024)

    recorder = ModeRecorder(model.momentum.U, Vector{Float64}[], Float64[])
    (; samples, times) = recorder
    fm = FieldModes(model.momentum.U; name="U_modes", frequency=frequency, modes=1:3)
    fm_p = FieldModes(model.momentum.p; name="p_modes", frequency=frequency, modes=[2, 1], update_interval=2*timestep)
    config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs,
        postprocess=[fm, fm_p, FieldAverage(model.momentum.U; name="U_mean"), recorder])

    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, 0.0)
    mktempdir() do dir
        cd(dir) do
            run!(model, config; progress=false)
            output = read("time_$(iterations)_postprocessed.vtk", String)
            # every field of every object is written
            for label ∈ ("U_modes_mean", "U_modes_m1_re", "U_modes_m3_im", "U_modes_m2_amp", "p_modes_mean", "p_modes_m2_amp", "U_mean")
                @test occursin(label, output)
            end
        end
    end

    # against the discrete Fourier sums of the recorded samples of the last whole period, in every cell
    window = 51:150
    coefficient(m) = sum(samples[i] .* cis(2π*m*frequency*times[i]) for i ∈ window) ./ length(window)
    for (k, m) ∈ enumerate(fm.modes)
        c = coefficient(m)
        @test fm.re[k].x.values ≈ real.(c) rtol = 1e-8
        @test fm.im[k].x.values ≈ imag.(c) rtol = 1e-8
        @test mode_amplitude(fm, m).x.values ≈ 2 .* abs.(c) rtol = 1e-8
        @test mode_amplitude(fm, m; rms=true).x.values ≈ sqrt(2) .* abs.(c) rtol = 1e-8
    end
    sample_mean = sum(samples[window]) ./ length(window)
    @test fm.mean.x.values ≈ sample_mean rtol = 1e-8
    @test_throws ArgumentError mode_amplitude(fm, 4)

    # the wave: u(t) = mean + Σ 2 Re(c_m e^{-imωt})
    t = times[120]
    wave = reconstruct_modes(fm, t; modes=[1, 3], mean=true)
    expected = sample_mean .+ sum(2 .* real.(coefficient(m) .* cis(-2π*m*frequency*t)) for m ∈ (1, 3))
    @test wave.x.values ≈ expected rtol = 1e-8
    reconstruct_modes!(wave, fm, t)
    @test wave.x.values ≈ sum(2 .* real.(coefficient(m) .* cis(-2π*m*frequency*t)) for m ∈ 1:3) rtol = 1e-8

    # the flow follows the inlet, U0(1 + A cos ωt): amplitude U0 A at the forcing frequency in the cells
    # next to the inlet, harmonics far below. Sanity bound, as in test_field_rms.jl
    inlet_cells = lastindex(fm.mean.x.values)-25:lastindex(fm.mean.x.values)-15
    @test sum(mode_amplitude(fm, 1).x.values[inlet_cells])/length(inlet_cells) ≈ U0*A rtol = 0.1
    @test maximum(mode_amplitude(fm, 3).x.values[inlet_cells]) < 0.05*U0*A
    @test any(!iszero, fm_p.re[1].values) # the scalar field, sampled every other step
end

@testset "window and checks" begin
    mesh = UNV2D_mesh(pkgdir(XCALibre, "examples/0_GRIDS", "quad40.unv"), scale=0.001)
    transient = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=1e-3), turbulence=RANS{Laminar}(),
        energy=Energy{Isothermal}(), domain=mesh)
    steady = Physics(time=Steady(), fluid=Fluid{Incompressible}(nu=1e-3), turbulence=RANS{Laminar}(),
        energy=Energy{Isothermal}(), domain=mesh)
    p = transient.momentum.p
    dt = 0.25
    window(fm, iterations) = (c = convert_time_to_iterations(fm, transient, dt, iterations); (c.start, c.stop, c.update_interval))

    # period 8 dt: of 30 steps the last 24; sampled every 2 dt from step 3, the 12 samples 7, 9, ... 29
    @test window(FieldModes(p; name="m", frequency=0.5, modes=1:2), 30) == (7, 30, 1)
    @test window(FieldModes(p; name="m", frequency=0.5, modes=1, start=3dt, update_interval=2dt), 30) == (7, 29, 2)
    @test window(FieldModes(p; name="m", frequency=0.5, modes=1, stop=20dt), 30) == (5, 20, 1)
    # less than a period, or a period that is not a whole number of samples: warned, not an error
    @test (@test_logs (:warn,) match_mode=:any window(FieldModes(p; name="m", frequency=0.5, modes=1), 5)) == (1, 5, 1)
    @test_logs (:warn,) match_mode=:any window(FieldModes(p; name="m", frequency=0.3, modes=1), 30)
    # above the Nyquist frequency of the samples
    @test_throws ArgumentError window(FieldModes(p; name="m", frequency=0.5, modes=1:4), 30)
    @test_throws ArgumentError convert_time_to_iterations(FieldModes(p; name="m", frequency=0.5, modes=1), steady, 1, 30)
    @test_throws ArgumentError FieldModes(p; name="m", frequency=0.5, modes=0:2)
    @test_throws ArgumentError FieldModes(p; name="m", frequency=-1, modes=1:2)
end

# A field built from known waves on a spanwise-extruded 3D mesh, stepped by hand
@testset "spanwise modes" begin
    mesh = UNV3D_mesh(pkgdir(XCALibre, "examples/0_GRIDS", "box32x32x32.unv"))
    model = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=1e-3), turbulence=RANS{Laminar}(),
        energy=Energy{Isothermal}(), domain=mesh)
    U = model.momentum.U
    centres = [c.centre for c ∈ mesh.cells]
    x, y, z = (getindex.(centres, i) for i ∈ 1:3)
    span = (maximum(z) - minimum(z))*32/31 # 32 cells
    β = 2π/span
    frequency = 2.0
    ω = 2π*frequency

    # amplitudes that vary over the plane, one wave per (m, n); (1, 1) and (1, -1) are an unequal oblique pair
    profile = @. (1 + x)*(2 + y^2)
    waves = Dict((2, 0) => (0.5, 0.3), (1, 1) => (0.2, 1.1), (1, -1) => (0.1, -0.4), (3, 2) => (0.05, 2.0))
    streak = (0.07, 0.6) # steady, wavenumber 2β
    u(t) = profile .* (1 .+ streak[1] .* cos.(2β .* z .- streak[2]) .+
        sum(A .* cos.(n*β .* z .- m*ω*t .- φ) for ((m, n), (A, φ)) ∈ waves))

    iterations, dt = 64, 1/(frequency*32) # two periods of 32 samples
    fm = FieldModes(U; name="U_modes", frequency=frequency, modes=1:3)
    stepping = convert_time_to_iterations(fm, model, dt, iterations)
    for iter ∈ 1:iterations
        U.x.values .= u(iter*dt)
        U.y.values .= 2 .* U.x.values
        runtime_postprocessing!(stepping, iter, iterations, nothing, iter*dt, nothing)
    end

    modes = [(2, 0), (1, 1), (1, -1), (3, 2), (0, 0), (0, 2), (2, 1)]
    sm = spanwise_modes(fm, mesh; wavenumber=β, modes=modes)
    # A cos(nβz - mωt - φ) is the mode (m, n) with c = (A/2) e^{-iφ}
    for ((m, n), (A, φ)) ∈ waves
        j = findfirst(==((m, n)), modes)
        @test sm.re[j].x.values ≈ profile .* (A/2*cos(φ)) rtol = 1e-10
        @test sm.im[j].x.values ≈ profile .* (-A/2*sin(φ)) rtol = 1e-10
        @test sm.re[j].y.values ≈ 2 .* sm.re[j].x.values rtol = 1e-10
        @test mode_amplitude(sm, (m, n)).x.values ≈ A .* profile rtol = 1e-10
        @test mode_amplitude(sm, (m, n); rms=true).x.values ≈ A/sqrt(2) .* profile rtol = 1e-10
    end
    @test mode_amplitude(sm, (0, 0)).x.values ≈ profile rtol = 1e-10
    @test mode_amplitude(sm, (0, 2)).x.values ≈ streak[1] .* profile rtol = 1e-10
    @test maximum(abs, mode_amplitude(sm, (2, 1)).x.values) < 1e-10*maximum(profile) # a mode that is not in the field
    @test all(iszero, mode_amplitude(sm, (1, 1)).z.values)

    # all the modes together are the field; one mode alone is its wave
    t = 0.123
    @test reconstruct_modes(sm, t).x.values ≈ u(t) rtol = 1e-10
    A, φ = waves[(1, -1)]
    @test reconstruct_modes(sm, t; modes=[(1, -1)]).x.values ≈ profile .* A .* cos.(-β .* z .- ω*t .- φ) rtol = 1e-10
    # and in time only: the mean and the three multiples carry the whole field
    @test reconstruct_modes(fm, t; mean=true).x.values ≈ u(t) rtol = 1e-10

    # a spanwise direction other than z
    along_x = convert_time_to_iterations(FieldModes(U; name="m", frequency=frequency, modes=1), model, dt, iterations)
    for iter ∈ 1:iterations
        @. U.x.values = (2 + y^2)*cos(β*x - ω*iter*dt)
        runtime_postprocessing!(along_x, iter, iterations, nothing, iter*dt, nothing)
    end
    sx = spanwise_modes(along_x, mesh; wavenumber=β, modes=[(1, 1)], direction=[1, 0, 0])
    @test mode_amplitude(sx, (1, 1)).x.values ≈ 2 .+ y.^2 rtol = 1e-10

    # the written fields
    names = first.(build_args(sm))
    @test "U_modes_m1_n-1_amp" ∈ names && "U_modes_m0_n2_re" ∈ names && length(names) == 3*length(modes)

    @test_throws ArgumentError spanwise_modes(fm, mesh; wavenumber=β, modes=[(4, 1)])  # multiple not extracted
    @test_throws ArgumentError spanwise_modes(fm, mesh; wavenumber=β, modes=[(0, -1)]) # conjugate of (0, 1)
    @test_throws ArgumentError spanwise_modes(fm, mesh; wavenumber=β, modes=[(1, 1)], direction=[1, 1, 0]) # no lines
    @test_logs (:warn,) match_mode=:any spanwise_modes(fm, mesh; wavenumber=1.3β, modes=[(1, 1)]) # span not whole wavelengths
end

end
