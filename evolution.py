import torch
import random 
import os
from dataclasses import dataclass
from typing import TypeVar, List, Type
from entity import Individual, Oasis, Position
import bisect
from dotenv import load_dotenv

    
load_dotenv()

MUTATION_RATE = float(os.getenv("MUTATION_RATE"))
CROSSOVER_RATE = float(os.getenv("CROSSOVER_RATE"))
SCREEN_WIDTH = int(os.getenv('SCREEN_WIDTH')) 
SCREEN_HEIGHT = int(os.getenv('SCREEN_HEIGHT'))
 
T = TypeVar("T")


class Evolution:    
    """
    evolution engine

    """

    @dataclass
    class Evo_Individual:
        individual: Individual
        fitness: float


    def __init__(
            self,
            population_size: int, 
            genome: Type[T], 
            individual_type: Type[T],
            resources: List[Oasis] = None
            ):
        
        self.best_individual = Evolution.Evo_Individual(genome(), 0.0)
        self.genome = genome
        self.individual_type = individual_type
        self.resources = resources
        self.population = self.init_population(population_size, self.individual_type, self.genome)
        self.maximize(self.population)
        

    def __repr__(self):
        out = f"evolution(population={self.population}, resources={self.resources})"
        return out 


    def init_population(self, pop_size: int, individual: Type[T], genome: Type[T]) -> List[T]: 

        population = [individual(Position(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)), genome) for i in range(pop_size)]
        out = [ Evolution.Evo_Individual(individual, self.get_fitness(individual)) for individual in population ]
        return out


    def maximize(self, individuals: List[Evo_Individual]):
        """ keep track of the best individual """
        
        for indiv in individuals: 
            if indiv.fitness > self.best_individual.fitness:
                self.best_individual = indiv



    def mutate(self, offspring: Evo_Individual) -> Evo_Individual:
        """
        mutation function for individuals to explore more possibilities
        """
        genome = offspring.individual.get_genome()
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
        offspring.individual.load_genome(new_genome)
        return offspring


    def create_position(self) -> Position: 
        
        while True: 
            pos_x = random.randint(0, SCREEN_WIDTH)
            pos_y = random.randint(0, SCREEN_HEIGHT)
            new_position = Position(pos_x, pos_y)
            stop = True
            for individual in self.population: 
                if new_position == individual.individual.position:
                    stop = False

            if stop:
                break

        return new_position


    def cross_over(
            self,
            parent_one: Evo_Individual,
            parent_two: Evo_Individual
            ) -> List[Evo_Individual]:
        """
        one point crossover function 
        """

        cross_over_probability = random.random()
        if(cross_over_probability > CROSSOVER_RATE):
            return [parent_one, parent_two]

        genome_one = parent_one.individual.get_genome()
        genome_two = parent_two.individual.get_genome()
        cross_over_point = random.randint(1, len(genome_one) - 1)
        offspring_one = self.individual_type(self.create_position())
        offspring_two = self.individual_type(self.create_position())
        offspring_one_genome = torch.cat((genome_one[:cross_over_point], genome_two[cross_over_point:]))
        offspring_two_genome = torch.cat((genome_two[:cross_over_point], genome_one[cross_over_point:]))
        offspring_one.load_genome(offspring_one_genome)
        offspring_two.load_genome(offspring_two_genome)
        out = [Evolution.Evo_Individual(offspring_one, self.get_fitness(offspring_one)), Evolution.Evo_Individual(offspring_two, self.get_fitness(offspring_two))]
        self.maximize(out)
        return out



    def step_evolution(self):
        """ creates the next generation """
        new_population = []
        luck_wheel = []
        s = 0.0
        for individual in self.population:
            s += individual.fitness
            luck_wheel.append(s)

        for _ in range(len(self.population) // 2):

            rnd = random.randint(0, int(luck_wheel[-1]))
            index_one = bisect.bisect_left(luck_wheel, rnd)
            index_two = 0
            while index_one == index_two:
                rnd = random.randint(0, int(luck_wheel[-1]))
                index_two = bisect.bisect_left(luck_wheel, rnd)

            parent_one = self.population[index_one]
            parent_two = self.population[index_two]

            offsprings = self.cross_over(parent_one, parent_two)
            offsprings = [self.mutate(offspring) for offspring in offsprings]
            new_population.extend(offsprings)
        
        self.population = new_population

       

    def get_distance(self, position_one: Position, position_two: Position) -> float: 
        """ returns the distance between two positions in the map """
        out = ((position_one.pos_x - position_two.pos_x)**2 + (position_one.pos_y - position_two.pos_y)**2)**(0.5) 
        return out


    def get_fitness(self, individual: Individual) -> float:
        """ calculates the fitness of the individual """
        out = 1_000_000_000
        for resource in self.resources:
            tmp = self.get_distance(individual.position, resource.position)
            out = min(tmp, out)
        out = (SCREEN_HEIGHT**2 + SCREEN_WIDTH**2)**(0.5) - out #to make sure that bigger fitness is better
        return out 