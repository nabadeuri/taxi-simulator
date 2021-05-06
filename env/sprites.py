import pygame
from env.settings import *


class Taxi(pygame.sprite.Sprite):
    def __init__(self):
        super(Taxi, self).__init__()
        self.surf = pygame.image.load("env/assets/taxi.png")
        self.surf.set_colorkey(BLACK)
        self.rect = self.surf.get_rect()


# passenger sprite
class Passenger(pygame.sprite.Sprite):
    def __init__(self):
        super(Passenger, self).__init__()
        self.surf = pygame.image.load("env/assets/passenger.png")
        self.surf.set_colorkey(BLACK)
        self.rect = self.surf.get_rect()


# destination sprite
class Destination(pygame.sprite.Sprite):
    def __init__(self):
        super(Destination, self).__init__()
        self.surf = pygame.image.load("env/assets/destination.png")
        self.surf.set_colorkey(BLACK)
        self.rect = self.surf.get_rect()


class Obstacle(pygame.sprite.Sprite):
    def __init__(self, width, height):
        super(Obstacle, self).__init__()
        self.surf = pygame.Surface((width, height))
        self.surf.fill((255, 0, 0))
        self.rect = self.surf.get_rect()


# road srpite
class Road(pygame.sprite.Sprite):
    def __init__(self):
        super(Road, self).__init__()
        self.surf = pygame.Surface((BOX_WIDTH, BOX_HEIGHT))
        self.surf.fill(BACKGROUND)
        self.rect = self.surf.get_rect()
        self.draw_border()

    def draw_border(self):
        pygame.draw.rect(self.surf, WHITE, self.rect, 1)
