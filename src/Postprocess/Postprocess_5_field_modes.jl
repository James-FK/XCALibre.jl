export FieldModes, SpanwiseModes
export spanwise_modes, reconstruct_modes, reconstruct_modes!, mode_amplitude

@kwdef struct FieldModes{T<:AbstractField,S<:AbstractString}
    field::T
    name::S
    frequency::Float64
    modes::Vector{Int}
    mean::T
    re::Vector{T}
    im::Vector{T}
    start::Union{Real,Nothing}
    stop::Union{Real,Nothing}
    update_interval::Union{Real,Nothing}
end

"""
    FieldModes(
    #required arguments
    field;
    name::String,
    frequency::Real,
    modes,

    #optional keyword arguments
    start::Union{Real,Nothing},
    stop::Union{Real,Nothing},
    update_interval::Union{Real,Nothing})

Constructor to allocate memory to store the Fourier coefficients in time of a field, in every cell,
at multiples of a base frequency. Once created, should be passed to the `Configuration` object as an
argument with keyword `postprocess`. Transient simulations only.

For each multiple `m` in `modes` the complex coefficient `c_m = re + i im` of `e^{-i m ω t}`,
`ω = 2π frequency`, is accumulated in every cell as the running mean of `field * e^{i m ω t}` over
the sampling window, with the time mean of the field alongside. In each cell the field is then

    field(t) ≈ mean + Σ_m 2 Re(c_m e^{-i m ω t})

so that the wave of mode `m` has the amplitude `2|c_m|` (rms `√2|c_m|`) and the phase `arg(c_m)`:
`A cos(m ω t - φ)` gives `c_m = (A/2) e^{iφ}`. `t` is the simulation time. See [`mode_amplitude`](@ref)
and [`reconstruct_modes`](@ref) for the amplitude and the wave of each mode after the run, and
[`spanwise_modes`](@ref) to split the coefficients further into spanwise wavenumbers.

# Input arguments
- `field` the `VectorField` or `ScalarField` to be decomposed, e.g , `model.momentum.U`.
- `name::String` the label of the output fields. The time mean is written as `name_mean`, and each mode as `name_m<m>_re`, `name_m<m>_im` and `name_m<m>_amp` (the amplitude `2|c_m|`).
- `frequency::Real` the base frequency, in Hz (cycles per unit of flow time).
- `modes` the integer multiples of `frequency` to extract, e.g. `1:5` or `[1, 2, 3, 5]`.

# Optional arguments
- `start::Union{Real,Nothing}` optional keyword which specifies the start of the sampling window, in **flow time**.
- `stop::Union{Real,Nothing}` optional keyword which specifies the end of the sampling window. Default value is the last timestep.
- `update_interval::Union{Real,Nothing}` optional keyword which specifies how often the field is sampled (default value is every timestep). Note that the frequency of writing the post-processed fields is specified by the `write_interval` in `Configuration`.

The modes are only orthogonal over whole periods of `frequency`, so the window is shortened to the
last whole number of periods that fit between `start` and `stop`, and the first samples are left out.
The time step must be fixed, with a whole number of samples in each period of `frequency`, and the
highest mode must be below the Nyquist frequency of the samples; a warning is given otherwise. Until
the window is complete the stored coefficients are those of the samples taken so far.
"""
function FieldModes(field; name::AbstractString, frequency::Real, modes,
    start::Union{Real,Nothing}=nothing, stop::Union{Real,Nothing}=nothing, update_interval::Union{Real,Nothing}=nothing)
    field isa Union{ScalarField,VectorField} || throw(ArgumentError("Unsupported field type: $(typeof(field))"))
    frequency > 0 || throw(ArgumentError("frequency must be > 0 (got $frequency)"))
    modes = Int[modes...]
    !isempty(modes) && all(>(0), modes) && allunique(modes) || throw(ArgumentError(
        "modes must be distinct positive multiples of the base frequency (got $modes); the time mean (mode 0) is always stored"))
    return FieldModes(field=field, name=name, frequency=Float64(frequency), modes=modes, mean=_similar_field(field),
        re=[_similar_field(field) for _ ∈ modes], im=[_similar_field(field) for _ ∈ modes],
        start=start, stop=stop, update_interval=update_interval)
end

_similar_field(field::ScalarField) = ScalarField(field.mesh)
_similar_field(field::VectorField) = VectorField(field.mesh)
_component_values(field::ScalarField) = (field.values,)
_component_values(field::VectorField) = (field.x.values, field.y.values, field.z.values)

function runtime_postprocessing!(fm::FieldModes, iter::Integer, n_iterations::Integer, Str, time, config)
    must_calculate(fm, iter, n_iterations) || return nothing
    n = div(iter - fm.start, fm.update_interval) + 1
    u = _component_values(fm.field)
    TF = eltype(u[1])
    a = one(TF)/n
    b = one(TF) - a
    for (mean, ui) ∈ zip(_component_values(fm.mean), u)
        @. mean = b*mean + a*ui
    end
    for (k, m) ∈ enumerate(fm.modes)
        sinθ, cosθ = sincos(2π*m*fm.frequency*Float64(time)) # the phase in Float64 whatever the field's
        ac, as = a*TF(cosθ), a*TF(sinθ)
        for (re, im, ui) ∈ zip(_component_values(fm.re[k]), _component_values(fm.im[k]), u)
            @. re = b*re + ac*ui
            @. im = b*im + as*ui
        end
    end
    return nothing
end

function convert_time_to_iterations(fm::FieldModes, model, dt, iterations)
    model.time === Transient() || throw(ArgumentError("FieldModes needs a transient simulation"))
    dt = Float64(dt)
    if fm.start === nothing
        start = 1
    else
        fm.start >= 0 || throw(ArgumentError("Start must be a value ≥ 0 (got $(fm.start))"))
        start = clamp(ceil(Int, fm.start/dt), 1, iterations)
    end
    if fm.stop === nothing
        stop = iterations
    else
        fm.stop ≥ 0 || throw(ArgumentError("stop must be ≥ 0 (got $(fm.stop))"))
        stop = clamp(floor(Int, fm.stop/dt), 1, iterations)
    end
    if fm.update_interval === nothing
        update_interval = 1
    else
        fm.update_interval > 0 || throw(ArgumentError("update interval must be > 0 (got $(fm.update_interval))"))
        update_interval = max(1, floor(Int, fm.update_interval/dt))
    end
    stop >= start || throw(ArgumentError("After conversion with dt=$dt the sampling window is empty (start = $start, stop = $stop)"))

    report = is_report_rank(fm.field.mesh)
    sample_dt = update_interval*dt
    2*maximum(fm.modes)*fm.frequency*sample_dt < 1 || throw(ArgumentError(
        "FieldModes $(fm.name): mode $(maximum(fm.modes)) (frequency $(maximum(fm.modes)*fm.frequency)) is not below the Nyquist frequency $(1/(2*sample_dt)) of the samples"))

    # the last whole periods of the base frequency between start and stop
    per_period = 1/(fm.frequency*sample_dt)
    available = div(stop - start, update_interval) + 1
    periods = floor(Int, available/per_period*(1 + 1e-6))
    if periods ≥ 1
        samples = round(Int, periods*per_period)
        report && abs(samples - periods*per_period) > 1e-6*samples && @warn "FieldModes $(fm.name): a period of the base frequency holds $per_period samples, not a whole number, so the modes leak into each other; choose the time step or update_interval so that it does"
        stop = start + (available - 1)*update_interval
        start = stop - (samples - 1)*update_interval
        report && @info "FieldModes $(fm.name): sampling $samples samples over $periods whole period(s) of the base frequency, from time $(start*dt) to $(stop*dt)"
    else
        report && @warn "FieldModes $(fm.name): the sampling window holds $available samples, less than one period of the base frequency ($per_period samples), so the coefficients are not those of the modes"
    end
    return FieldModes(field=fm.field, name=fm.name, frequency=fm.frequency, modes=fm.modes, mean=fm.mean, re=fm.re, im=fm.im,
        start=start, stop=stop, update_interval=update_interval)
end

# Output fields: the mean and, per mode, the real and imaginary parts and the amplitude of the wave
function build_args(fm::FieldModes)
    args = Any[("$(fm.name)_mean", fm.mean)]
    for (k, m) ∈ enumerate(fm.modes)
        label = "$(fm.name)_m$(m)"
        push!(args, (label*"_re", fm.re[k]), (label*"_im", fm.im[k]), (label*"_amp", mode_amplitude(fm, m)))
    end
    return Tuple(args)
end

# |c| scaled to the amplitude of the wave: a steady mode is its own coefficient, the others add their conjugate
function _amplitude!(out, re, im, steady, rms)
    scale = steady ? 1 : rms ? sqrt(2) : 2
    for (o, a, b) ∈ zip(_component_values(out), _component_values(re), _component_values(im))
        TF = eltype(o)
        @. o = TF(scale)*sqrt(a^2 + b^2)
    end
    return out
end

_mode_index(modes, mode) = begin
    k = findfirst(==(mode), modes)
    k === nothing && throw(ArgumentError("mode $mode was not extracted (available: $modes)"))
    k
end

"""
    mode_amplitude(fm::FieldModes, m; rms=false)
    mode_amplitude(sm::SpanwiseModes, (m, n); rms=false)

The amplitude of a mode in every cell, as a new field of the type of the decomposed field (one
amplitude per component for a `VectorField`): `2|c|`, the peak of the wave `2 Re(c e^{iθ})`, or its
rms `√2|c|` with `rms=true`. For the steady modes of a `SpanwiseModes`, `(0, 0)` gives `|c|` whatever
`rms`, as it is not a wave.

A pair of oblique waves `(m, n)` and `(m, -n)` forms a standing pattern across the span whose
amplitude at the antinodes is the sum of the two amplitudes.
"""
function mode_amplitude(fm::FieldModes, m::Integer; rms=false)
    k = _mode_index(fm.modes, m)
    _amplitude!(_similar_field(fm.field), fm.re[k], fm.im[k], false, rms)
end

"""
    reconstruct_modes(fm::FieldModes, time; modes=fm.modes, mean=false)
    reconstruct_modes(sm::SpanwiseModes, time; modes=sm.modes)

The field of the chosen modes at `time`, in every cell, as a new field of the type of the decomposed
field: for a `FieldModes` the sum of the waves `2 Re(c_m e^{-i m ω t})` of the multiples `m` in
`modes`, plus the time mean with `mean=true`; for a `SpanwiseModes` the sum of the waves
`2 Re(c_mn e^{i(n β s - m ω t)})` of the pairs `(m, n)` in `modes` (`(0, 0)` contributes `c_00`).
`reconstruct_modes!(out, ...)` writes into an existing field, e.g. to write a sequence of times.

The result can be written with `write_results`, as any other field.
"""
reconstruct_modes(fm::FieldModes, time; kwargs...) = reconstruct_modes!(_similar_field(fm.field), fm, time; kwargs...)

"""
    reconstruct_modes!(out, fm::FieldModes, time; modes=fm.modes, mean=false)
    reconstruct_modes!(out, sm::SpanwiseModes, time; modes=sm.modes)

As [`reconstruct_modes`](@ref), writing into the existing field `out`.
"""
function reconstruct_modes!(out, fm::FieldModes, time; modes=fm.modes, mean=false)
    outs = _component_values(out)
    TF = eltype(outs[1])
    for (o, mean_i) ∈ zip(outs, _component_values(fm.mean))
        mean ? (o .= mean_i) : (o .= zero(TF))
    end
    for m ∈ modes
        k = _mode_index(fm.modes, m)
        sinθ, cosθ = sincos(2π*m*fm.frequency*Float64(time))
        c, s = 2*TF(cosθ), 2*TF(sinθ)
        for (o, re, im) ∈ zip(outs, _component_values(fm.re[k]), _component_values(fm.im[k]))
            @. o += c*re + s*im
        end
    end
    return out
end

struct SpanwiseModes{T<:AbstractField,S<:AbstractString,C<:AbstractField}
    name::S
    frequency::Float64
    wavenumber::Float64
    modes::Vector{Tuple{Int,Int}}
    re::Vector{T}
    im::Vector{T}
    span::C # spanwise coordinate of every cell
end

"""
    spanwise_modes(fm::FieldModes, mesh; wavenumber, modes, direction=[0, 0, 1], tol=nothing)

Split the coefficients of a [`FieldModes`](@ref) into spanwise wavenumbers, after the run. For each
pair `(m, n)` in `modes` the coefficient `c_mn` of `e^{i(n β s - m ω t)}` is the mean over each
spanwise line of cells of `c_m e^{-i n β s}`, with `β = wavenumber`, `ω = 2π fm.frequency` and `s` the
coordinate along `direction`. Returns a `SpanwiseModes`, which holds `c_mn = re + i im` in every
cell (the same value in all the cells of a line) for [`mode_amplitude`](@ref) and
[`reconstruct_modes`](@ref), and whose fields are written with `write_results(..., build_args(sm)...)`.

# Input arguments
- `fm` the `FieldModes` holding the coefficients in time.
- `mesh` the mesh object stored on the CPU, for the cell centres. On a distributed mesh pass the distributed mesh (on the CPU) on every rank.

# Input keyword arguments
- `wavenumber` the base spanwise wavenumber `β = 2π/λ`, in radians per unit length.
- `modes` the pairs `(m, n)`: `m` a multiple of the base frequency extracted by `fm`, or `0` for the time mean, and `n` the multiple of `wavenumber`, of either sign. With `m ≥ 1`, `(m, n)` and `(m, -n)` are the two oblique waves, travelling to either side. With `m = 0` they are conjugates of each other, so only `n ≥ 0` is accepted.

# Optional arguments
- `direction` the spanwise direction (default `[0, 0, 1]`).
- `tol` the distance below which two cell centres are on the same spanwise line, measured normal to `direction` (default: `1e-4` of the cube root of the smallest cell volume).

The mesh must be extruded along `direction` with a uniform spacing, so that every cell belongs to a
line of the same number of cells; the span must hold a whole number of wavelengths of every mode
(a warning is given otherwise). On a distributed mesh the lines may cross the partition: the
coefficients are gathered over the ranks, so each rank holds arrays of the size of the whole mesh
during the call.
"""
function spanwise_modes(fm::FieldModes, mesh; wavenumber::Real, modes, direction=[0, 0, 1], tol=nothing)
    modes = [(Int(m), Int(n)) for (m, n) ∈ modes]
    allunique(modes) || throw(ArgumentError("modes must be distinct (got $modes)"))
    for (m, n) ∈ modes
        m == 0 || m ∈ fm.modes || throw(ArgumentError("mode ($m, $n): the multiple $m of the base frequency was not extracted (available: 0 and $(fm.modes))"))
        m == 0 && n < 0 && throw(ArgumentError("mode ($m, $n): a steady mode (0, -n) is the conjugate of (0, n), ask for n ≥ 0"))
    end
    pmesh = fm.field.mesh
    β = Float64(wavenumber)
    e = normalize(SVector{3,Float64}(direction))
    e1 = normalize(cross(e, abs(e[1]) < 0.9 ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)))
    e2 = cross(e, e1)

    # mesh-wide arrays, in the order of the undecomposed mesh; each rank contributes its own cells
    n_owned = n_owned_cells(pmesh)
    n_cells = global_sum(n_owned, pmesh)
    owned_ids = [Int(global_cell_index(pmesh, i)) for i ∈ 1:n_owned]
    gather(values) = begin
        all_cells = zeros(Float64, n_cells)
        all_cells[owned_ids] .= @view values[1:n_owned]
        global_sum(all_cells, pmesh)
    end
    centres = [SVector{3,Float64}(mesh.cells[i].centre) for i ∈ 1:n_owned]
    s = gather([c⋅e for c ∈ centres])
    if tol === nothing
        tol = 1e-4*cbrt(global_min(minimum(Float64(mesh.cells[i].volume) for i ∈ 1:n_owned; init=Inf), pmesh))
    end
    line, n_lines = _spanwise_lines(gather([c⋅e1 for c ∈ centres]), gather([c⋅e2 for c ∈ centres]), tol)
    per_line = _check_spanwise_lines(line, n_lines, s, β, unique(abs(n) for (_, n) ∈ modes), is_report_rank(pmesh))

    sm = SpanwiseModes(fm.name, fm.frequency, β, modes, [_similar_field(fm.field) for _ ∈ modes],
        [_similar_field(fm.field) for _ ∈ modes], ScalarField(pmesh))
    local_ids = [Int(global_cell_index(pmesh, i)) for i ∈ eachindex(sm.span.values)] # ghosts included
    scatter!(values, all_cells) = copyto!(values, eltype(values).(all_cells[local_ids]))
    scatter!(sm.span.values, s)

    line_re = zeros(n_lines)
    line_im = zeros(n_lines)
    for m ∈ unique(first.(modes))
        k = m == 0 ? 0 : _mode_index(fm.modes, m)
        re_fields = _component_values(m == 0 ? fm.mean : fm.re[k])
        im_fields = m == 0 ? map(_ -> nothing, re_fields) : _component_values(fm.im[k])
        for (component, (re_field, im_field)) ∈ enumerate(zip(re_fields, im_fields))
            a = gather(Array(re_field))
            b = im_field === nothing ? zeros(n_cells) : gather(Array(im_field))
            for (j, (mj, n)) ∈ enumerate(modes)
                mj == m || continue
                line_re .= 0
                line_im .= 0
                for i ∈ 1:n_cells # (a + ib) e^{-inβs}
                    sinθ, cosθ = sincos(n*β*s[i])
                    line_re[line[i]] += a[i]*cosθ + b[i]*sinθ
                    line_im[line[i]] += b[i]*cosθ - a[i]*sinθ
                end
                scatter!(_component_values(sm.re[j])[component], line_re[line] ./ per_line)
                scatter!(_component_values(sm.im[j])[component], line_im[line] ./ per_line)
            end
        end
    end
    return sm
end

# Index of the spanwise line of every cell, from the two coordinates p and q normal to the span:
# cells are on the same line when both are within tol
function _spanwise_lines(p, q, tol)
    cluster(v) = begin
        id = zeros(Int, length(v))
        k = 0
        previous = -Inf
        for i ∈ sortperm(v)
            v[i] - previous > tol && (k += 1)
            id[i] = k
            previous = v[i]
        end
        id
    end
    p_id, q_id = cluster(p), cluster(q)
    lines = Dict{Tuple{Int,Int},Int}()
    line = [get!(lines, (p_id[i], q_id[i]), length(lines) + 1) for i ∈ eachindex(p)]
    return line, length(lines)
end

# Every line must hold the same number of cells; warns if the first line is not uniformly spaced or
# the span is not a whole number of wavelengths of the wavenumbers. Returns the cells per line.
function _check_spanwise_lines(line, n_lines, s, β, wavenumbers, report)
    counts = zeros(Int, n_lines)
    for l ∈ line
        counts[l] += 1
    end
    per_line = counts[1]
    all(==(per_line), counts) || throw(ArgumentError(
        "the cells do not form spanwise lines of equal length (found $n_lines lines of $(minimum(counts)) to $(maximum(counts)) cells): the mesh must be extruded along `direction`; if it is, adjust `tol`"))
    per_line > 1 || throw(ArgumentError("found one cell per spanwise line: the mesh must be extruded along `direction`; if it is, adjust `tol`"))
    report || return per_line
    first_line = sort(s[findall(==(1), line)])
    spacing = diff(first_line)
    mean_spacing = (first_line[end] - first_line[1])/(per_line - 1)
    maximum(abs, spacing .- mean_spacing) < 1e-3*mean_spacing || @warn "spanwise_modes: the cells are not uniformly spaced along the span, so the modes leak into each other"
    span = per_line*mean_spacing
    for n ∈ wavenumbers
        waves = n*β*span/(2π)
        abs(waves - round(waves)) < 1e-3 || @warn "spanwise_modes: the span $span holds $waves wavelengths of wavenumber $(n*β), not a whole number, so the modes leak into each other"
    end
    return per_line
end

function mode_amplitude(sm::SpanwiseModes, mode::Tuple{Integer,Integer}; rms=false)
    j = _mode_index(sm.modes, (Int(mode[1]), Int(mode[2])))
    _amplitude!(_similar_field(sm.re[j]), sm.re[j], sm.im[j], mode == (0, 0), rms)
end

reconstruct_modes(sm::SpanwiseModes, time; kwargs...) = reconstruct_modes!(_similar_field(sm.re[1]), sm, time; kwargs...)

function reconstruct_modes!(out, sm::SpanwiseModes, time; modes=sm.modes)
    outs = _component_values(out)
    TF = eltype(outs[1])
    foreach(o -> o .= zero(TF), outs)
    s = sm.span.values
    for mode ∈ modes
        m, n = Int(mode[1]), Int(mode[2])
        j = _mode_index(sm.modes, (m, n))
        scale = TF((m, n) == (0, 0) ? 1 : 2)
        nβ = TF(n*sm.wavenumber)
        θt = 2π*m*sm.frequency*Float64(time)
        sinθt, cosθt = TF.(sincos(θt)) # θ = nβs - θt, split so that the time enters in Float64
        for (o, re, im) ∈ zip(outs, _component_values(sm.re[j]), _component_values(sm.im[j]))
            # Re((re + i im) e^{iθ}) = re cos θ - im sin θ
            @. o += scale*(re*(cos(nβ*s)*cosθt + sin(nβ*s)*sinθt) - im*(sin(nβ*s)*cosθt - cos(nβ*s)*sinθt))
        end
    end
    return out
end

function build_args(sm::SpanwiseModes)
    args = Any[]
    for (j, (m, n)) ∈ enumerate(sm.modes)
        label = "$(sm.name)_m$(m)_n$(n)"
        push!(args, (label*"_re", sm.re[j]), (label*"_im", sm.im[j]), (label*"_amp", mode_amplitude(sm, (m, n))))
    end
    return Tuple(args)
end
