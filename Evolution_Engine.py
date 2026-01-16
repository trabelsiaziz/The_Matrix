from dotenv import load_dotenv
import random
from typing import TypeVar, List, Type
from Entity import individual
import torch
import os

load_dotenv()

POPULATION_SIZE = int(os.getenv("POPULATION_SIZE"))
WATER_SOURCE = int(os.getenv("WATER_SOURCE"))
MUTATION_RATE = float(os.getenv("MUTATION_RATE"))
CROSSOVER_RATE = float(os.getenv("CROSSOVER_RATE"))
GENERATION_LIMIT = int(os.getenv("GENERATION_LIMIT"))

T = TypeVar("T")

def init_population(pop_size: int, indiv: Type[T]) -> List[T]: 
    
    population = [indiv() for i in range(pop_size)]
    return population


def mutate(offspring: individual) -> individual:
    
    genome = offspring.get_genome()
    new_genome = torch.Tensor()
    for gene in genome: 
        mutation_probability = random.random()
        
    

