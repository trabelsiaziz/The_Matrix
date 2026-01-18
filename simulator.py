# This class serves only as a renderer to the environment !!
from environment import Environment
import pygame


class Simulation: 
    def __init__(self, environment: Environment):
        self.environment = environment
    
    def __repr__(self):
        out = f"Simulation(environment={self.environment})"
        return out
    
    def render(self):
        pygame.init()
        screen = pygame.display.set_mode((self.environment.width, self.environment.height))
        clock = pygame.time.Clock()
        running = True
        dt = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                    
            screen.fill("yellow")
            
            for individual in self.environment.population:
                individual_position = pygame.Vector2(individual.position.pos_x, individual.position.pos_y)
                pygame.draw.circle(screen, "brown", individual_position, 20)
            
            for resource in self.environment.resources:
                resource_position = pygame.Vector2(resource.position.pos_x, resource.position.pos_y)
                pygame.draw.circle(screen, "blue", resource_position, 20)
            
            self.environment.step()           
            pygame.display.flip()
            dt = clock.tick(60) / 1000

        pygame.quit()
