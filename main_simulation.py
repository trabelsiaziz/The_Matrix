from simulator import Simulation
from environment import Environment
from entity import Organism, Oasis, Position
from evolution import init_population

pop = init_population(5, Organism)
env = Environment(population=pop, resources=[Oasis(position=Position(767, 342))])
sim = Simulation(env)

sim.render()