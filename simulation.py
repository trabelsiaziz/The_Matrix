from simulator import Simulation
from environment import Environment
from entity import Organism, Oasis, Position
from evolution import init_population
from dotenv import load_dotenv
import os 


load_dotenv()
POPULATION_SIZE = int(os.getenv("POPULATION_SIZE"))
WATER_SOURCE = int(os.getenv('WATER_SOURCE'))

pop = init_population(POPULATION_SIZE, Organism)
env = Environment(population=pop, resources=[Oasis(position=Position(767, 342))])
sim = Simulation(env)

sim.render()