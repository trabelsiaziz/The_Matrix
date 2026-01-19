import torch
from torch import nn
from dotenv import load_dotenv
import os
from abc import ABC, abstractmethod
from typing import TypeVar, Type


load_dotenv()
POPULATION_SIZE = int(os.getenv("POPULATION_SIZE"))
WATER_SOURCE = int(os.getenv('WATER_SOURCE'))

 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = TypeVar('T', nn.Module, list) # supports torch neural networks and simple lists as genomes

class Position:
    
    def __init__(self, pos_x: int = None, pos_y: int = None):
        self.pos_x = pos_x 
        self.pos_y = pos_y
    
    def __repr__(self):
        out = f"Position(x={self.pos_x if self.pos_x is not None else -1}, y={self.pos_y if self.pos_y is not None else -1})"
        return out


class Individual(ABC):
    """ Abstract class for all individuals """
    def __init__(self, genome: Type[T], position: Position = None):
        if not isinstance(genome, (nn.Module, list)):
            raise TypeError(f"genome must be nn.Module or list, got {type(genome)}")
        self.genome = genome
        self.position = position

    @abstractmethod
    def get_genome(self) -> torch.Tensor:
        pass 

    @abstractmethod
    def load_genome(self, new_genome: torch.Tensor): 
        pass

    @abstractmethod
    def take_action(self) -> int: 
        pass
    

F_IN = (WATER_SOURCE + POPULATION_SIZE + 1) * 2
# +1 for the env dimensions

class Brain(nn.Module): 
    """ The brain of the organism (the genome) """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(F_IN, 64, dtype=float, device=device)
        self.fc2 = nn.Linear(64, 32, dtype=float, device=device)
        self.fc3 = nn.Linear(32, 4, dtype=float, device=device)
        self.out = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.out(x)
        return torch.argmax(x, dim=-1)
        



class Organism(Individual): 
    
    def __init__(self, position: Position = None):
        super().__init__(Brain())
        self.position = position
    
    def __repr__(self):
        out = f"organism(x={self.position.pos_x if self.position is not None else -1}, y={self.position.pos_y if self.position is not None else -1})"
        return out
    
    def get_genome(self) -> torch.Tensor: 
        """Convert all parameters of the model into a flat vector"""
        params = []
        for param in self.genome.parameters():
            params.append(param.data.view(-1))  
        params = torch.cat(params)
        return params
    
    def load_genome(self, params: torch.Tensor) -> None: 
        """Load a flat vector into the model"""
        pointer = 0
        for param in self.genome.parameters():
            num_param = param.numel()  # total elements in this layer
            # slice from vector and reshape
            param.data.copy_(params[pointer:pointer + num_param].view_as(param))
            pointer += num_param
    

    def take_action(self, input_state: torch.tensor) -> int:
        """ returns action taken by the Organism """
        return self.genome(input_state)
    
    
        
        

class Oasis: 
    def __init__(self, position: Position = None):
        self.position = position

    def __repr__(self):
        out = f"Oasis(x={self.position.pos_x if self.position is not None else -1}, y={self.position.pos_y if self.position is not None else -1})"
        return out
    