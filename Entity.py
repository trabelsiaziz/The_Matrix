import torch
from torch import nn
from dotenv import load_dotenv
import os
import random
from abc import ABC, abstractmethod
from typing import TypeVar, Type, List


load_dotenv()
width = int(os.getenv("SCREEN_WIDTH"))
height = int(os.getenv("SCREEN_HEIGHT"))
X = width * height

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = TypeVar('T', nn.Module, list) # supports torch neural networks and simple lists as genomes


class individual(ABC):
    """ Abstract class for all individuals """
    def __init__(self, genome: Type[T]):
        if not isinstance(genome, (nn.Module, list)):
            raise TypeError(f"genome must be nn.Module or list, got {type(genome)}")
        self.genome = genome

    @abstractmethod
    def get_genome(self):
        pass 

    @abstractmethod
    def load_genome(self): 
        pass



class Brain(nn.Module): 
    def __init__(self):
        super().__init__()
        self.observe = nn.Linear(2, 4, dtype=float, device=device)
        

    def forward(self, x):
        x = self.observe(x)
        return x
        




class Organism(individual): 
    
    def __init__(self, pos_x: int, pos_y: int):
        super().__init__(Brain())
        self.pos_x = pos_x
        self.pos_y = pos_y
    
    def __repr__(self):
        out = f"This is a basic organism which is currently located at (x={self.pos_x}, y={self.pos_y})"
        return out
    
    def get_genome(self) -> torch.Tensor: 
        """Convert all parameters of the model into a flat vector"""
        params = []
        for param in self.genome.parameters():
            params.append(param.data.view(-1))  # flatten each tensor
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
        
    
    

class Oasis: 
    def __init__(self):
        self.pos_x = random.randint(1, width - 1)
        self.pos_y = random.randint(1, height - 1)

    def __repr__(self):
        out = f"This is an oasis which is currently located at (x={self.pos_x}, y={self.pos_y})"
        return out
    