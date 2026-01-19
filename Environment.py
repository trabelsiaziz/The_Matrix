from entity import Individual, Oasis, Position, Organism
from evolution import step_evolution
from typing import List 
from dotenv import load_dotenv
import random
import torch
import os


load_dotenv()

SCREEN_WIDTH = int(os.getenv('SCREEN_WIDTH')) 
SCREEN_HEIGHT = int(os.getenv('SCREEN_HEIGHT'))
POPULATION_SIZE = int(os.getenv("POPULATION_SIZE"))
WATER_SOURCE = int(os.getenv('WATER_SOURCE'))
EVOLUTION_RATE = int(os.getenv('EVOLUTION_RATE'))

TIME = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
F_IN = (WATER_SOURCE + POPULATION_SIZE + 1) * 2

class Environment: 
    
    def __init__(self, population: List[Individual] = None, resources: List[Oasis] = None):
        self.population = population
        self.resources = resources
        self.height = SCREEN_HEIGHT
        self.width = SCREEN_WIDTH

    
    def __repr__(self):
        out = f"Environment(population={self.population}, resources={self.resources})"
        return out
    
    def create_position(self) -> Position: 
        
        while True: 
            pos_x = random.randint(0, self.width)
            pos_y = random.randint(0, self.height)
            new_position = Position(pos_x, pos_y)
            stop = True
            for individual in self.population: 
                if new_position == individual.position:
                    stop = False

            if stop:
                break

        return new_position
    
    
    
    def step(self):
        """ evolve the environment """
        global TIME

        if TIME == EVOLUTION_RATE:
            #print("step_evolution is called !!")
            TIME %= EVOLUTION_RATE
            new_population = step_evolution(self.population, self.resources, Organism)
            self.population = []
            for individual in new_population:
                individual.position = self.create_position()
            self.population = new_population
            assert(len(self.population) == POPULATION_SIZE)

        individual_positions = [individual.position for individual in self.population]
        resources_positions = [resource.position for resource in self.resources]
        
        for individual in self.population:
            other_positions = [pos for pos in individual_positions if pos != individual.position]
            input_data = [coord for pos in other_positions for coord in [pos.pos_x, pos.pos_y]]
            input_data = [self.height, self.width] + [coord for pos in resources_positions for coord in [pos.pos_x, pos.pos_y]] + [individual.position.pos_x, individual.position.pos_y] + input_data
            input_data = torch.tensor(input_data, dtype=float, device=device)
            
            tmp = F_IN - input_data.shape[0]
            #assert(tmp == 0)
            
            input_data = torch.cat([input_data, torch.zeros(tmp, dtype=float, device=device)])
            action = individual.take_action(input_data)
            
            match action:
                case 0: #move up 
                    individual.position.pos_y = min(individual.position.pos_y + 1, self.height)
                case 1: #move down
                    individual.position.pos_y = max(individual.position.pos_y - 1, 0)
                case 2: #move right
                    individual.position.pos_x = min(individual.position.pos_x + 1, self.width)
                case 3: #move left
                    individual.position.pos_x = max(individual.position.pos_x - 1, 0)
                case _:
                    pass 
        TIME += 1
        

        
        
