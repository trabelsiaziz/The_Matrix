from entity import Individual, Oasis, Position, Organism
from evolution import Evolution
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
    
    def __init__(self, evolution: Evolution):
        
        self.evolution = evolution
        self.height = SCREEN_HEIGHT
        self.width = SCREEN_WIDTH

    
    def __repr__(self):
        out = f"Environment(evolution={self.evolution})"
        return out
    
    
    
    def step(self) -> bool:
        """ evolve the environment """
        global TIME
        update_gen_count = False

        if TIME == EVOLUTION_RATE:
            update_gen_count = True
            TIME %= EVOLUTION_RATE
            self.evolution.step_evolution()
            # self.population = []
            # for individual in new_population:
            #     individual.position = self.create_position()
            # self.population = new_population
            # assert(len(self.population) == POPULATION_SIZE) 

        individual_positions = [individual.individual.position for individual in self.evolution.population]
        resources_positions = [resource.position for resource in self.evolution.resources]
        
        for individual in self.evolution.population:
            other_positions = [pos for pos in individual_positions if pos != individual.individual.position]
            input_data = [coord for pos in other_positions for coord in [pos.pos_x, pos.pos_y]]
            input_data = [self.height, self.width] + [coord for pos in resources_positions for coord in [pos.pos_x, pos.pos_y]] + [individual.individual.position.pos_x, individual.individual.position.pos_y] + input_data
            input_data = torch.tensor(input_data, dtype=float, device=device)
            
            tmp = F_IN - input_data.shape[0]
            #assert(tmp == 0)
            
            input_data = torch.cat([input_data, torch.zeros(tmp, dtype=float, device=device)])
            action = individual.individual.take_action(input_data)
            
            match action:
                case 0: #move up 
                    individual.individual.position.pos_y = min(individual.individual.position.pos_y + 1, self.height)
                case 1: #move down
                    individual.individual.position.pos_y = max(individual.individual.position.pos_y - 1, 0)
                case 2: #move right
                    individual.individual.position.pos_x = min(individual.individual.position.pos_x + 1, self.width)
                case 3: #move left
                    individual.individual.position.pos_x = max(individual.individual.position.pos_x - 1, 0)
                case _:
                    pass 
        TIME += 1
        return update_gen_count
        

        
        
