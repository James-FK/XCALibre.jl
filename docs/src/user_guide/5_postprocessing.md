# Post-processing
*Information about postprocessing XCALibre.jl results*

## ParaView
---

All solvers in XCALibre.jl will write simulation results in formats that can be loaded directly in [ParaView](https://www.paraview.org/), which is the leading open-source project for scientific visualisation and postprocessing. More information about how to use [ParaView](https://www.paraview.org/) can be found in the [resources](https://www.paraview.org/resources/) page on their website.

XCALibre.jl can output simulation results to either `VTK` compliant formats or `OpenFOAM` format. XCALibre.jl uses two different `VTK` formats depending on the type of flow solver used. For 2D simulations, the results are written to file using the `.vtk` file format. 3D simulations are written using the unstructured `VTK` file format, `.vtu`. 

!!! note

    A limitation of the current `VTK` writers in XCALibre.jl is that boundary information is stored along with internal mesh cell information, and results are stored at cell centres only. Thus, care must be taken when visualising results at boundary faces. Boundary information for `fixedValue` boundaries is displayed corrently when the results are saved in the `OpenFOAM` format. 

## Available functions

Although [ParaView](https://www.paraview.org/) offers considerable flexibility for postprocessing results, users may also wish to carry out more advanced or different analyses on their CFD results. At present XCALibre.jl offers a limited set of built-in runtime postprocessing functions, currently these include time averaging a scalar or vector field over a specified range of iterations, and also the root mean square (RMS) of the fluctuations of a field. 
### Example: Calculate runtime averaged field 
As an example, to average any Scalar or Vector field an instance of `FieldAverage` must be created. For transient simulations this is a time average and for steady simulations it is an average over iterations. 
```@docs; canonical=false
FieldAverage
```
Once created this is simply passed to the `Configuration` object as an extra argument with the keyword `postprocess`. For example to average the velocity field over the whole simulation, 
```julia
postprocess = FieldAverage(model.momentum.U; name = "U_mean")
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs, postprocess=postprocess)
```
The rest of the case would remain exactly the same. 
### Example: Calculate field RMS 
The RMS of a Scalar of Vector field can be obtained in a similar way to the time averaged field, instead an instance of `RMS` is created which has the following definition. 
```@docs; canonical=false
FieldRMS
```
The RMS of the velocity field can be easily calculated by creating an instance of `FieldRMS` and passing it to the `Configuration` object with the keyword `postprocess`, with the rest of the case remaining unchanged.
```julia
postprocess = FieldRMS(model.momentum.U; name = "U_rms")
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs, postprocess=postprocess)
```
## Example: Calculate the Reynolds Stress Tensor
The Reynolds Stress Tensor can be obtained using the constructor `ReynoldsStress`, which has to be passed to the `Configuration` object. 
```@docs; canonical=false
ReynoldsStress
```
## Example: Calculate the Q-criterion
The Q-criterion can be obtained using the constructor `Qcriterion`, which has to be passed to the `Configuration` object. 
```@docs; canonical=false
Qcriterion
```
## Example: Put a probe at a location in the compuational domain and output to file
To create a probe, before creating the `Configuration` object, create an instance of `Probe`.
```@docs; canonical=false
Probe
```
For example to probe velocity at location `[1.0, 2.0, 3.0]`:
```julia
postprocess = Probe(model.momentum.U, mesh; location=[1.0, 2.0, 3.0], name="velocity probe at [1,2,3]")
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs, postprocess=postprocess)
```
On a distributed mesh pass the distributed mesh and create the probe on every rank; the rank that
owns the nearest cell samples it and rank 0 writes the file.

### Example: Extract the modes of a periodic flow

For a flow that is periodic in time, such as a forced boundary layer, `FieldModes` accumulates the Fourier coefficients of a field in every cell, at multiples of a base frequency.
```@docs; canonical=false
FieldModes
```
For example, for a boundary layer forced at the frequency `f` whose subharmonic `f/2` is also of interest, the base frequency is `f/2`, and the multiples 1 to 5 are `f/2`, `f`, `3f/2`, `2f` and `5f/2`:
```julia
modes = FieldModes(model.momentum.U; name="U_modes", frequency=f/2, modes=1:5, start=t_developed)
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs, postprocess=modes)
```
The mean, the real and imaginary parts of each coefficient and the amplitude of each mode are written with the other post-processed fields. After the run, the amplitude and the wave of any mode are available in every cell:
```@docs; canonical=false
mode_amplitude
reconstruct_modes
```
```julia
A_f = mode_amplitude(modes, 2; rms=true)         # rms amplitude of the fundamental, a VectorField
wave = reconstruct_modes(modes, time; modes=[1]) # the subharmonic wave at one time
```
On a mesh extruded along the span, the coefficients can be split further into spanwise wavenumbers, giving the modes `(m, n)`, of frequency `m` times the base frequency and spanwise wavenumber `n` times a base wavenumber:
```@docs; canonical=false
spanwise_modes
```
```julia
# the 2D fundamental, the two oblique subharmonic waves and the steady streaks of the same wavenumber
spanwise = spanwise_modes(modes, mesh; wavenumber=β, modes=[(2, 0), (1, 1), (1, -1), (0, 1)])
A_11 = mode_amplitude(spanwise, (1, 1))
oblique = reconstruct_modes(spanwise, time; modes=[(1, 1), (1, -1)]) # the standing pattern of the pair
```

### Example: Post-process multiple fields 
To post-process multiple fields as a time, a vector of objects can be passed instead e.g. 
```julia
postprocess = [FieldRMS(model.momentum.U; name = "U_rms"), FieldAverage(model.momentum.U; name = "U_mean"), FieldAverage(model.momentum.p; name = "p_mean")]
config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs, postprocess=postprocess)
```


If more functionality is required, defining new custom postprocessing functions is reasonably straight-forward since these can be written in pure Julia. In this section, examples of postprocessing functions will be provided as an illustration. 

!!! note

    At present the postprocessing functions available in XCALibre.jl shown below will only execute on CPUs and should be considered experimental. Once we settle on a "sensible" (maintainable and extensible) API, we plan to offer a larger selection of postprocessing tools which are likely to include more options for runtime postprocessing.

### Example: Calculate boundary average

In this example, a function is shown that can be used to calculate the average on a user-provided boundary. 

```@docs; canonical=false
boundary_average
```

To calculate pressure and viscous forces, the following functions are available:

```@docs; canonical=false
pressure_force
viscous_force
```

### Example: Volume integrals and averages

Volume integrals of scalar and vector fields run on the field's backend (CPU or GPU).

```@docs; canonical=false
volume_integral
weighted_volume_integral
volume_average
total_volume
```

For example, the volume-averaged velocity of a model is `volume_average(model.momentum.U, config)`.
