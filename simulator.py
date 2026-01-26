# This class serves only as a renderer to the environment !!
from environment import Environment
import pygame


class Simulation: 

    def __init__(self, environment: Environment):
        self.environment = environment
        self.gen_count = 0
    
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
            
            # Display generation count
            font = pygame.font.Font(None, 36)
            text = font.render(f"Generation: {self.gen_count}", True, "black")
            screen.blit(text, (10, 10))
            
            for individual in self.environment.evolution.population:
                individual_position = pygame.Vector2(individual.individual.position.pos_x, individual.individual.position.pos_y)
                if individual == self.environment.evolution.best_individual:
                    pygame.draw.circle(screen, "green", individual_position, 20)
                else:
                    pygame.draw.circle(screen, "brown", individual_position, 20)
            
            for resource in self.environment.evolution.resources:
                resource_position = pygame.Vector2(resource.position.pos_x, resource.position.pos_y)
                pygame.draw.circle(screen, "blue", resource_position, 20)
            
            update = self.environment.step()  
            if update:
                self.gen_count += 1   


            pygame.display.flip()
            dt = clock.tick(60) / 1000
        pygame.quit()
    