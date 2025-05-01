import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

"""
Units used in the simulation:
    [x] = electron Debeye length, lambda_D
    [t] = inverse (angular) plasma frequency w_p,e^-1
    [v] = electron thermal speed, vth,e
    [density] = q_e * n_e,0
    [E] = m_e * v_th,e * w_p,e / q_e

In these units, Gauss's law is \grad E = rho.
"""

class Species:
    """
    Represents a plasma species in the Particle-In-Cell simulation.
    
    Attributes:
        name:           Name of the species
        q:              Charge of the macroparticle
        m:              Mass of the macroparticle
        N_particles:    Number of macroparticles
        x:              Particle positions (initialized uniformly over the grid)
        vx:             Particle velocities (initialized with a drift and thermal spread)
        vhalfx:         Half-step velocities for leapfrog integration
        Ex_particles:   Interpolated electric field at particle positions
    """
    def __init__(self, name: str, q: int, m: float, N_particles: int, vth: float, vdrift: float, Lx: float):
        self.name = name
        self.q = q
        self.m = m
        self.N_particles = N_particles

        # Initialize particle positions uniformly over the grid
        self.x = np.linspace(0.0, Lx, N_particles, endpoint=False)
        # Initialize velocities with a thermal distribution about the drift velocity
        self.vx = np.random.normal(vdrift, vth, N_particles)
        self.vhalfx = np.zeros_like(self.vx)  # For leapfrog scheme

        # Placeholder for the electric field interpolated at particle positions
        self.Ex_particles = np.zeros(N_particles)

def charge_deposition(sp: Species, grid: np.ndarray, charge_density: np.ndarray) -> None:
    """Deposit particle charge onto the grid using the Cloud-in-Cell (CIC) method"""

    dx = grid[1] - grid[0]
    N_grid = len(grid)
    
    for i in range(sp.N_particles):
        xi = sp.x[i]
        # Determine indices of the left and right grid points
        idx_left = int(np.floor(xi / dx)) % N_grid
        idx_right = (idx_left + 1) % N_grid

        # Fraction of distance from the left grid point
        delta_x = (xi - grid[idx_left]) / dx

        # Deposit 1-delta_x to left grid node, the rest to right node
        weight_left = 1.0 - delta_x
        weight_right = delta_x
        
        charge_density[idx_left] += sp.q * weight_left / dx
        charge_density[idx_right] += sp.q * weight_right / dx

def field_solve(N_grid: int, dx: float, charge_density: np.ndarray) -> np.ndarray:
    """Solve for the electric field on the grid by solving Poisson's equation"""

    # Set up the discrete Laplacian
    main_diag = -2*np.ones(N_grid)
    off_diag = np.ones(N_grid-1)
    laplacian = diags([off_diag, main_diag, off_diag], offsets=[-1,0,1], shape=(N_grid, N_grid), format="lil")
    
    # Apply periodic boundary conditions
    laplacian[0,-1] = 1
    laplacian[-1,0] = 1
    laplacian = laplacian.tocsr()

    # Enforce zero-mean charge density for periodic Poisson solve
    charge_density -= np.mean(charge_density)

    # Solve discretized Poisson's equation: d^2(phi)/dx^2 = -rho
    phi = spsolve(laplacian, -dx**2 * charge_density)

    # Construct a matrix for computing the electric field from the potential
    derivative = diags([-off_diag, off_diag], offsets=[-1,1], shape=(N_grid, N_grid), format="lil")
    
    # Periodic boundary conditions
    derivative[0,-1] = -1
    derivative[-1,0] = 1
    derivative = derivative.tocsr()
    
    # Compute electric field, E = -dphi/dx, using central differencing
    Ex_grid = -(derivative @ phi) / (2*dx)
    
    return Ex_grid

def field_interpolation(sp: Species, grid: np.ndarray, Ex_grid: np.ndarray) -> np.ndarray:
    """Interpolate the electric field from the grid to the particles using the CIC method"""

    dx = grid[1] - grid[0]
    N_grid = len(grid)
    Ex_particles = np.zeros(sp.N_particles)

    for i in range(sp.N_particles):
        xi = sp.x[i]
        idx_left = int(np.floor(xi / dx)) % N_grid
        idx_right = (idx_left + 1) % N_grid

        delta_x = (xi - grid[idx_left]) / dx
        weight_left = 1.0 - delta_x
        weight_right = delta_x
        
        Ex_particles[i] = Ex_grid[idx_left] * weight_left + Ex_grid[idx_right] * weight_right

    return Ex_particles

def compute_energies(species: list, Ex_grid: np.ndarray, dx: float):
    """Compute kinetic, field, and total energies at each timestep"""
    
    kinetic_energy = sum(0.5 * sp.m * np.sum(sp.vx**2) for sp in species)
    field_energy = 0.5 * np.sum(Ex_grid**2) * dx
    total_energy = kinetic_energy + field_energy
    return total_energy, kinetic_energy, field_energy


class PICSimulation:
    """Class that encapsulates the Particle-In-Cell simulation"""

    def __init__(self, charge_neutralizing_background: bool = False):

        # Simulation grid and time parameters
        self.N_cells = 100
        self.particles_per_cell = 100
        self.N_particles = int(self.particles_per_cell * self.N_cells)  # per species
        print("N_particles per species:", self.N_particles)
        self.N_grid = self.N_cells  # for periodic BCs
        self.Lx = 5.0  # grid length
        self.grid = np.linspace(0.0, self.Lx, self.N_grid, endpoint=False)
        self.dx = self.grid[1] - self.grid[0]
        self.N_timesteps = 750
        self.dt = 1e-3

        # Define the ion (singly ionized) background charge density if needed
        self.charge_neutralizing_background = charge_neutralizing_background

        # Initialize energy history arrays
        self.total_energy_history = np.zeros(self.N_timesteps)
        self.kinetic_energy_history = np.zeros(self.N_timesteps)
        self.field_energy_history = np.zeros(self.N_timesteps)

        # For phase-space plotting
        self.jitter = np.random.uniform(-0.05, 0.05, size=self.N_particles)

        self.create_initial_conditions()
        self.validate_parameters()

    def create_initial_conditions(self):
        """Set initial conditions for the simulation"""
        
        # Electron species parameters
        self.vdrift_e = 3.0
        self.vth_e = 1.0

        # Create two electron species with opposite drift
        self.electrons1 = Species('e- 1', -1, 1.0, self.N_particles, self.vth_e, self.vdrift_e, self.Lx)
        self.electrons2 = Species('e- 2', -1, 1.0, self.N_particles, self.vth_e, -self.vdrift_e, self.Lx)
        self.species = [self.electrons1, self.electrons2]

        if self.charge_neutralizing_background:
            self.N_ions = sum(sp.N_particles for sp in self.species if sp.q < 0)
            self.ion_charge_density = (-1) * self.electrons1.q * self.N_ions / self.Lx

        # Sinusoidally perturb the electrons to kickstart the two-stream instability
        delta = 0.01 * self.Lx
        kx = 2 * np.pi / self.Lx

        for sp in self.species:
            sp.x += delta * np.sin(kx * sp.x)
            sp.x %= self.Lx  # apply periodic BC

    def validate_parameters(self):
        """Ensures simulation parameters are instantiated properly for numerical stability"""

        net_charge = sum(sp.q * sp.N_particles for sp in self.species)
        if self.charge_neutralizing_background:
            net_charge += self.N_ions
        assert np.isclose(net_charge, 0), "Net charge must be zero for periodic Poisson solve"

        CFL = 1.0 / (np.abs(self.vdrift_e) + 3*self.vth_e)
        assert self.dt < CFL, f"CFL condition dt = {self.dt} < {CFL} not satisfied"
        assert self.dt < 0.1, f"Timestep {self.dt} must be less than 0.1 to resolve plasma period"
        assert self.dx < 1.0, f"Grid spacing {self.dx} must be less than 1.0 to resolve Debeye length"

    def plot_phase_space(self, ts: int):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Top subplot: Phase space
        ax1.set_title(f"Time = {ts*self.dt:.2f} $\omega_{{pe}}^{{-1}}$\n\nPhase Space")
        for sp in self.species:
            highlight_idx = len(sp.x) // 2
            
            # Create a mask for the normal points (all except the one to highlight)
            normal_mask = np.ones(len(sp.x), dtype=bool)
            normal_mask[highlight_idx] = False
            ax1.scatter(sp.x[normal_mask], sp.vx[normal_mask], label=sp.name, s=0.1)
            ax1.scatter(sp.x[highlight_idx], sp.vx[highlight_idx], 
                        label=f"{sp.name} (tracked)", s=50, marker='x', color='red')
        ax1.legend(loc='upper right')
        ax1.set_xlabel('x')
        ax1.set_ylabel('v$_x$')
        ax1.set_xlim([0, self.Lx])
        ax1.set_ylim([-10, 10])
        
        # Bottom subplot: Position space
        ax2.set_title("Position Space")
        offsets = [1/2, -1/2] # Vertical offsets for different species when plotting
        
        for i, sp in enumerate(self.species):
            # Create a slight vertical jitter for visual clarity
            y_positions = np.full_like(sp.x, offsets[i]) + self.jitter
            
            # Determine the index to highlight, here chosen to be the middle particle
            highlight_idx = len(sp.x) // 2
            
            # Create a mask for the normal points (all except the one to highlight)
            normal_mask = np.ones(len(sp.x), dtype=bool)
            normal_mask[highlight_idx] = False
            
            # Plot the normal points
            ax2.scatter(sp.x[normal_mask], y_positions[normal_mask], 
                        label=f"{sp.name} (all particles)", s=0.1)
            
            # Plot the highlighted particle with a distinct marker and color
            ax2.scatter(sp.x[highlight_idx], y_positions[highlight_idx], 
                        label=f"{sp.name} (tracked)", s=50, marker='x', color='red')
        
        ax2.legend(loc='upper right')
        ax2.set_xlabel('x')
        ax2.set_yticks([])  # Remove y-axis ticks as the vertical position is arbitrary
        ax2.set_xlim([0, self.Lx])
        ax2.set_ylim([-1, 1])
        
        plt.tight_layout()
        
        filename = f"plots/phasespace/phasespace_ts{str(ts).zfill(int(np.log10(self.N_timesteps)) + 1)}.jpg"
        plt.savefig(filename, dpi=200)
        plt.clf()

    def run(self):

        for ts in range(self.N_timesteps):
            print(f"--- Timestep {ts} ---")

            # Optionally plot phase space
            # if ts % (self.N_timesteps // 10) == 0:
            # if ts % 5 == 0:
            #     self.plot_phase_space(ts)

            # TIME INTEGRATION: Kick (half step update for velocities)
            for sp in self.species:
                sp.vhalfx = sp.vx + (sp.q / sp.m) * sp.Ex_particles * self.dt / 2.0

            # TIME INTEGRATION: Drift (update positions)
            for sp in self.species:
                sp.x += sp.vhalfx * self.dt
                sp.x %= self.Lx # periodic BCs

            # CHARGE DEPOSITION; deposit particle charge onto the grid
            charge_density = np.zeros_like(self.grid)
            for sp in self.species:
                charge_deposition(sp, self.grid, charge_density)

            # If background ions are included, add their contribution to the charge density
            if self.charge_neutralizing_background:
                charge_density += self.ion_charge_density

            net_charge_grid = np.sum(charge_density * self.dx)
            print(f"Net charge on grid: {net_charge_grid}")

            # FIELD SOLVE: Solve Poisson's equation to get the grid electric field
            Ex_grid = field_solve(self.N_grid, self.dx, charge_density)

            # FIELD INTERPOLATION: Interpolate grid E-field to particle positions
            for sp in self.species:
                sp.Ex_particles = field_interpolation(sp, self.grid, Ex_grid)

            # TIME INTEGRATION: Kick (complete velocity update)
            for sp in self.species:
                sp.vx = sp.vhalfx + (sp.q / sp.m) * sp.Ex_particles * self.dt / 2.0

            E_total, E_kin, E_field = compute_energies(self.species, Ex_grid, self.dx)
            self.total_energy_history[ts] = E_total
            self.kinetic_energy_history[ts] = E_kin
            self.field_energy_history[ts] = E_field
            print(f"Total Energy: {E_total:.4f}, Kinetic: {E_kin:.4f}, Field: {E_field:.4f}")

        plt.figure()
        plt.semilogy(self.kinetic_energy_history, label='Kinetic Energy')
        plt.semilogy(self.field_energy_history, label='Field Energy')
        plt.semilogy(self.total_energy_history, label='Total Energy', linestyle='--')
        plt.xlabel('Timestep')
        plt.ylabel('Energy')
        plt.title('Energy Conservation')
        plt.legend()
        plt.grid()
        plt.savefig("plots/energy_conservation.png")

def main():
    sim = PICSimulation(charge_neutralizing_background=True)
    sim.run()

if __name__ == "__main__":
    main()
