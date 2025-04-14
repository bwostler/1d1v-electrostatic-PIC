import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

"""
Code units:
[x] = electron Debeye length
[t] = electron plasma period
[v] = electron thermal speed
[density] = q_e * n_e,0
[E] = m_e * v_th,e * w_p,e / q_e

In these units, Gauss's law is \grad E = -rho.
"""

class Species:
    def __init__(
        self,
        name: str, # name of species
        q: int, # charge of macroparticle
        m: float, # mass of macroparticle
        N_particles : int, # number of macroparticles
        vth: float, # thermal speed of macroparticle
        vdrift: float, # drift velocity of macroparticle
        Lx: float # grid length along x
    ):
        self.name = name
        self.q = q
        self.m = m
        self.N_particles = N_particles

        # Basic initialization of species; uniformly distributed over grid,
        # and has a thermal speed distribution with a drift velocity
        self.x = np.linspace(0.0, Lx, N_particles, endpoint=False) # np.random.uniform(0.0, Lx, N_particles) 
        self.vx = np.random.normal(vdrift, vth, N_particles)
        self.vhalfx = np.zeros_like(self.vx) # for leapfrog

        # Field interpolated to particles from grid
        self.Ex_particles = np.zeros(N_particles)

def charge_deposition(sp: Species, grid: np.ndarray, charge_density: np.ndarray) -> None:
    """Charge deposition using the Cloud-in-Cell (CIC) method"""

    dx = grid[1] - grid[0]
    N_grid = len(grid)

    for i in range(sp.N_particles):
        xi = sp.x[i]

        # Get the grid point indices to the left and right of the particle
        idx_left = int(np.floor(xi / dx))
        idx_right = idx_left + 1

        # Handle periodic boundary conditions
        idx_left = idx_left % N_grid
        idx_right = idx_right % N_grid

        # Number of dx to left grid point from particle
        delta_x = (xi - grid[idx_left]) / dx

        # Assign 1-delta_x to that point (CIC)
        weight_left = 1.0 - delta_x

        # Do the same for the right, 1-(1-delta_x) = delta_x
        weight_right = delta_x

        charge_density[idx_left] += sp.q * weight_left / dx
        charge_density[idx_right] += sp.q * weight_right / dx

# Solve for the electric field on the grid using Poisson's equation
def field_solve(N_grid: int, dx: float, charge_density: np.ndarray) -> np.ndarray:
    
    # Construct periodic tridiagonal matrix A
    main_diag = -2*np.ones(N_grid)
    off_diag = np.ones(N_grid-1)

    # Initial tridiagonal (without terms for periodic BCs)
    laplacian = diags([off_diag,main_diag,off_diag], offsets=[-1,0,1], shape=(N_grid,N_grid), format="lil")

    # Add periodic BC terms
    laplacian[0,-1] = 1
    laplacian[-1,0] = 1
    laplacian = laplacian.tocsr()

    # For periodic BCs, a solution only exists if \int rho dx = 0, i.e. <rho>=0.
    # So we can subtract that from the charge desnity at no cost, but it *ensures*
    # that the constraint is satisfied.
    charge_density -= np.mean(charge_density)

    # Solve discretization of d^2 phi/dx^2 = -rho
    phi = spsolve(laplacian, -dx**2 * charge_density)

    # Compute E field from potential using central difference for dphi/dx
    derivative = diags([-off_diag,off_diag], offsets=[-1,1], shape=(N_grid,N_grid), format="lil")
    derivative[0,-1] = -1
    derivative[-1,0] = 1
    derivative = derivative.tocsr()
    
    # Solve discretization of E = -dphi/dx
    Ex_grid = -(derivative @ phi) / (2*dx) 

    return Ex_grid

def field_interpolation(sp: Species, grid: np.ndarray, Ex_grid: np.ndarray) -> np.ndarray:
    """Field interpolation using the Cloud-in-Cell (CIC) method"""

    Ex_particles = np.zeros(sp.N_particles)
    dx = grid[1] - grid[0]
    N_grid = len(grid)

    for i in range(sp.N_particles):
        xi = sp.x[i]
        idx_left = int(np.floor(xi / dx))
        idx_right = idx_left + 1

        # Handle periodic boundary conditions
        idx_left = idx_left % N_grid
        idx_right = idx_right % N_grid

        delta_x = (xi - grid[idx_left]) / dx
        weight_left = 1.0 - delta_x
        weight_right = delta_x
        
        Ex_particles[i] = Ex_grid[idx_left] * weight_left + Ex_grid[idx_right] * weight_right

    return Ex_particles

# For diagnostic purposes
def compute_energies(species: list, Ex_grid: np.ndarray, dx: float):
    kinetic_energy = 0.0
    for sp in species:
        kinetic_energy += 0.5 * sp.m * np.sum(sp.vx**2)

    field_energy = 0.5 * np.sum(Ex_grid**2) * dx

    total_energy = kinetic_energy + field_energy
    return total_energy, kinetic_energy, field_energy

def main(charge_neutralizing_background: bool = False):

    # Simulation parameters
    N_cells = 100
    particles_per_cell = 100
    N_particles = int(particles_per_cell * N_cells) # for each species
    N_grid = N_cells # equal when we use periodic BCs
    Lx = 5 # multiple of 2pi to capture full perturbation of electron positions
    grid = np.linspace(0.0, Lx, N_grid, endpoint=False)
    dx = grid[1]-grid[0]
    N_timesteps = 1000
    dt = 1e-3
    vdrift_e = 3
    vth_e = 1

    print("N_particles:", N_particles)

    electrons1 = Species('e- 1', -1, 1.0, N_particles, vth_e, vdrift_e, Lx)
    electrons2 = Species('e- 2', -1, 1.0, N_particles, vth_e, -vdrift_e, Lx)
    species = [electrons1, electrons2]

    N_ions = np.sum([sp.N_particles for sp in species if sp.q < 0])
    ion_charge_density = (-1) * electrons1.q * N_ions / Lx

    # Do various checks of the simulation parameters
    assert np.isclose(
        np.sum([sp.q * sp.N_particles for sp in species]) + int(charge_neutralizing_background) * N_ions, 0
    ), "Net charge must be zero for periodic Poisson solve"
    CFL = 1.0 / (np.abs(vdrift_e) + 3*vth_e)
    assert dt < CFL, f"CFL condition dt = {dt} < {CFL} not satisfied"
    assert dt < 0.1, f"Timestep {dt} must be less than 0.1 to resolve plasma period"
    assert dx < 1.0, f"Grid spacing {dx} must be less than 1.0 to resolve Debeye length"

    # Introduce small perturbation in electron positions as initial condition
    delta = 0.01 * Lx
    kx = 2 * np.pi / Lx
    electrons1.x += delta * np.sin(kx * electrons1.x)
    electrons1.x %= Lx # Periodic BC
    electrons2.x += delta * np.sin(kx * electrons2.x)
    electrons2.x %= Lx # Periodic BC

    total_energy_history = np.zeros(N_timesteps)
    kinetic_energy_history = np.zeros(N_timesteps)
    field_energy_history = np.zeros(N_timesteps)

    plt.figure()

    # PIC loop
    for ts in range(N_timesteps):

        print(f"timestep {ts}")

        if ts % (N_timesteps // 10) == 0:
            plt.figure()
            plt.title(f"timestep = {ts}")
            plt.scatter(electrons1.x, electrons1.vx, label=electrons1.name, s=0.5)
            plt.scatter(electrons2.x, electrons2.vx, label=electrons2.name, s=0.5)
            plt.legend(loc='upper right')
            plt.xlabel('x')
            plt.ylabel('vx')
            plt.xlim([0, Lx])
            plt.ylim([-10, 10])
            plt.savefig(f"plots/phasespace/phasespace_ts{str(ts).zfill(int(np.log10(N_timesteps))+1)}.png")
            plt.clf()

        # TIME INTEGRATION: Kick
        for sp in species:
            sp.vhalfx = sp.vx + (sp.q / sp.m) * sp.Ex_particles * dt / 2.0

        # TIME INTEGRATION: Drift
        for sp in species:
            sp.x += sp.vhalfx * dt
            sp.x %= Lx # Periodic BC

        # CHARGE DEPOSITION; deposit charge to grid

        # Initialize charge density to zero
        charge_density = np.zeros_like(grid)
        for sp in species:
            charge_deposition(sp, grid, charge_density)

        # Add ion background charge density
        if charge_neutralizing_background:
            charge_density += ion_charge_density

        # Testing that net charge is close to zero; very important for stability
        print("Net charge on grid:", np.sum(charge_density * dx))

        # FIELD SOLVE; solve for E field on grid
        Ex_grid = field_solve(N_grid, dx, charge_density)

        # FIELD INTERPOLATION; Interpolate E field from grid back to particles
        for sp in species:
            sp.Ex_particles = field_interpolation(sp, grid, Ex_grid)

        # TIME INTEGRATION: Kick
        for sp in species:
            sp.vx = sp.vhalfx + (sp.q / sp.m) * sp.Ex_particles * dt / 2.0

        E_total, E_kin, E_field = compute_energies(species, Ex_grid, dx)
        total_energy_history[ts] = E_total
        kinetic_energy_history[ts] = E_kin
        field_energy_history[ts] = E_field

        print(f"Total energy: {E_total:.4f}, Kinetic energy: {E_kin:.4f}, Field energy: {E_field:.4f}")

        # plt.title(f"timestep = {ts}")
        # plt.scatter(electrons.x, 0.5*np.ones(shape=electrons.x.shape), label=electrons.name, s=1)
        # plt.xlim([0,Lx])
        # plt.ylim([0,1])
        # plt.legend()
        # plt.savefig(f"plots/electrons_x_ts{ts}.png")
        # plt.clf()
        
        # plt.figure()
        # plt.hist(
        #     electrons.vx,
        #     bins=100,
        #     label=electrons.name,
        #     alpha=0.5
        # )
        # plt.legend()
        # plt.savefig(f"plots/electrons_vx_ts{ts}.png")
        # plt.clf()

    # plt.figure()
    # plt.semilogy(kinetic_energy_history, label='Kinetic Energy')
    # plt.semilogy(field_energy_history, label='Field Energy')
    # plt.semilogy(total_energy_history, label='Total Energy', linestyle='--')
    # plt.xlabel('Timestep')
    # plt.ylabel('Energy')
    # plt.title('Energy Conservation')
    # plt.legend()
    # plt.grid()
    # plt.savefig("plots/energy_conservation.png")

if __name__ == "__main__":
    main(charge_neutralizing_background=True)
