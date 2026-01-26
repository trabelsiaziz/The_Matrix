from simulator import Simulation
from evolution import Evolution
from environment import Environment
from entity import Organism, Oasis, Position, Brain
from dotenv import load_dotenv
import os 


load_dotenv()
POPULATION_SIZE = int(os.getenv("POPULATION_SIZE"))
WATER_SOURCE = int(os.getenv('WATER_SOURCE'))
OASIS_X = int(os.getenv('OASIS_X'))
OASIS_Y = int(os.getenv('OASIS_Y'))


resources = [Oasis(position=Position(OASIS_X, OASIS_Y))]
evol = Evolution(POPULATION_SIZE, Brain, Organism, resources)

env = Environment(evolution=evol)
sim = Simulation(env)

sim.render()