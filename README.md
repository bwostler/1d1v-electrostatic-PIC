# A simple 1d1v particle-in-cell code

A simple Python code to explore the basics of a 1d1v PIC code. The initial conditions are set to induce the two-stream instability between two counter-propagating electron beams. 

![Two-stream instability in phase and position space](docs/two-stream-instability.gif)

## Specifics of the implementation:
- Time integration is accomplished via leapfrog in its kick-drift-kick formulation.
- Charge deposition + field interpolation use Cloud-in-cell.
- The scalar potential is found by solving a discretized Poisson's equation, from which the electric field is computed by central differencing $E_x = -d\phi/dx$.