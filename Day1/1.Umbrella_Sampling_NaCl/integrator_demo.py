import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from openmm.app.topology import Topology
from openmm.app.element import Element
from openmm.app.forcefield import ForceField
from openmm.app.simulation import Simulation
from openmm.openmm import CustomExternalForce, LangevinIntegrator, LangevinMiddleIntegrator,\
    VariableLangevinIntegrator, NoseHooverIntegrator, BrownianIntegrator,\
    VerletIntegrator, Vec3
from openmm.unit.quantity import Quantity
import openmm.unit as unit
import openmm

if __name__ == '__main__':
    # parameter setup
    temperature = 300
    damping = 100
    timestep = 0.001
    steps = 10000
    start = Quantity(value=[Vec3(0.05, 0, 0)], unit=unit.nanometer)
    x_force_constant_per_two = 5000     # unit kJ/molnm^2
    # dummy system setup
    elem = Element(0, "X", "X", 1.0 * unit.amu)
    top = Topology()
    top.addChain()
    top.addResidue("xxx", top._chains[0])
    top.addAtom("X", elem, top._chains[0]._residues[0])
    ff = ForceField("./Day1/1.Umbrella_Sampling_NaCl/1d_pot.xml")     #("1d_pot.xml")
    # potential setup
    x_pot = CustomExternalForce(f"{x_force_constant_per_two} * x^2")
    y_pot = CustomExternalForce("100000 * y^2")
    z_pot = CustomExternalForce("100000 * z^2")
    
    #create openmm system
    # system = ff.createSystem(top) #denes solution, somehow wrong.
    system = openmm.System()
    mass = 1
    system.addParticle(mass)
    
    x_pot.addParticle(0)
    y_pot.addParticle(0)
    z_pot.addParticle(0)
    system.addForce(x_pot)
    system.addForce(y_pot)
    system.addForce(z_pot)
    # true potential representation
    x = np.linspace(-0.05, 0.05, 500)
    utotx = x_force_constant_per_two * x * x
    # listing integrators
    labels = [
        "verlet",
        # "Nose-Hoover",
        "langevin",
        # "var_langevin",
        "mid_langevin",
        "brownian"
    ]
    integrators = [
        VerletIntegrator(timestep),
        # NoseHooverIntegrator(temperature, damping, timestep),
        LangevinIntegrator(temperature, damping, timestep),
        # VariableLangevinIntegrator(temperature, damping, timestep),
        LangevinMiddleIntegrator(temperature, damping, timestep),
        BrownianIntegrator(temperature, damping, timestep)
    ]
    # preparing plots
    f, axes = plt.subplots(nrows=round(len(integrators) / 2), ncols=2)
    # calculating exact Boltzmann probability
    prob = np.exp(-utotx / (temperature * 0.0083144621)) / np.sum(np.exp(-utotx / (temperature * 0.0083144621)))
    for r in axes:
        for a in r:
            a.plot(x, prob)
    # running simulations
    for i in range(len(integrators)):
        sim = Simulation(top, system, integrators[i])
        sim.context.setPositions(start)
        dat = np.empty(shape=(steps, 8))
        for j in range(steps):
            state = sim.context.getState(getVelocities=True, getPositions=True, getEnergy=True)
            dat[j, 0:3] = state.getPositions(asNumpy=True)[0]._value
            dat[j, 3:6] = state.getVelocities(asNumpy=True)[0]._value
            dat[j, 6] = state.getPotentialEnergy()._value
            dat[j, 7] = state.getKineticEnergy()._value
            sim.step(1)
        # creating probability of the simulation
        simulation_dist = np.histogram(dat[:, 0], x)[0]
        axes[int(i / 2), i % 2].plot(x[1:], simulation_dist / np.sum(simulation_dist))
        axes[int(i / 2), i % 2].title.set_text(labels[i])
    f.suptitle(f"T={temperature:d} damping={damping:d} V={x_force_constant_per_two}x^2")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"Day1/1.Umbrella_Sampling_NaCl/{temperature:d}_{damping:d}_{x_force_constant_per_two}x^2.png")
