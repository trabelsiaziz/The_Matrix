import torch
import random 
import os
from typing import TypeVar, List, Type
from entity import Individual, Oasis, Position
import bisect
from dotenv import load_dotenv

    
load_dotenv()

MUTATION_RATE = float(os.getenv("MUTATION_RATE"))
CROSSOVER_RATE = float(os.getenv("CROSSOVER_RATE"))
SCREEN_WIDTH = float(os.getenv('SCREEN_WIDTH')) 
SCREEN_HEIGHT = float(os.getenv('SCREEN_HEIGHT'))
 
T = TypeVar("T")

def init_population(pop_size: int, individual: Type[T]) -> List[T]: 

    population = [individual(Position(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT))) for i in range(pop_size)]
    return population


def mutate(offspring: Individual) -> Individual:
    """
    mutation function for individuals to explore more possibilities
    """
    genome = offspring.get_genome()
    new_genome = []
    for gene in genome: 
        mutation_probability = random.random()
        if(mutation_probability <= MUTATION_RATE):
            tmp = random.random() 
            if tmp >= 0.5:
                gene += 0.01
            else:
                gene -= 0.01
        new_genome.append(gene)
    new_genome = torch.tensor(new_genome)
    offspring.load_genome(new_genome)
    return offspring


def cross_over(
        parent_one: Individual,
        parent_two: Individual,
        individual_type: Type[T]
        ) -> List[Individual]:
    """
    one point crossover function 
    """
    
    cross_over_probability = random.random()
    if(cross_over_probability > CROSSOVER_RATE):
        return [parent_one, parent_two]

    genome_one = parent_one.get_genome()
    genome_two = parent_two.get_genome()
    cross_over_point = random.randint(1, len(genome_one) - 1)
    offspring_one = individual_type()
    offspring_two = individual_type()
    offspring_one_genome = torch.cat((genome_one[:cross_over_point], genome_two[cross_over_point:]))
    offspring_two_genome = torch.cat((genome_two[:cross_over_point], genome_one[cross_over_point:]))
    offspring_one.load_genome(offspring_one_genome)
    offspring_two.load_genome(offspring_two_genome)
    return [offspring_one, offspring_two]



def step_evolution(population: List[Individual], resources: List[Oasis], individual_type: Type[T]) -> List[Individual]:
    """ creates the next generation """
    new_population = []
    luck_wheel = []
    s = 0.0
    for individual in population:
        s += fitness(individual, resources)
        luck_wheel.append(s)

    for _ in range(len(population) // 2):

        rnd = random.randint(0, int(luck_wheel[-1]))
        index_one = bisect.bisect_left(luck_wheel, rnd)
        index_two = 0
        while index_one == index_two:
            rnd = random.randint(0, int(luck_wheel[-1]))
            index_two = bisect.bisect_left(luck_wheel, rnd)
        
        parent_one = population[index_one]
        parent_two = population[index_two]
        
        offsprings = cross_over(parent_one, parent_two, individual_type)
        offsprings = [mutate(offspring) for offspring in offsprings]
        new_population.extend(offsprings)

    return new_population

def get_distance(position_one: Position, position_two: Position) -> float: 
    """ returns the distance between two positions in the map """
    out = ((position_one.pos_x - position_two.pos_x)**2 + (position_one.pos_y - position_two.pos_y)**2)**(0.5) 
    return out


def fitness(individual: Individual, resources: List[Oasis]) -> float:
    """ calculates the fitness of the individual """
    out = 1_000_000_000
    for resource in resources:
        tmp = get_distance(individual.position, resource.position)
        out = min(tmp, out)
    out = (SCREEN_HEIGHT**2 + SCREEN_WIDTH**2)**(0.5) - out #to make sure that bigger fitness is better
    return out 