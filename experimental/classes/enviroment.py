import random
import pygame
from utils.constants import *

class Environment:
    def __init__(self):
        self.food = set((random.randint(0, WORLD_COLS - 1), random.randint(0, WORLD_ROWS - 1)) for _ in range(200))

    def spawn_food(self):
        while len(self.food) < 200:
            self.food.add((random.randint(0, WORLD_COLS - 1), random.randint(0, WORLD_ROWS - 1)))

    def render(self, camera_x, camera_y):
        for fx, fy in self.food:
            screen_x = (fx * GRID_SIZE) - camera_x
            screen_y = (fy * GRID_SIZE) - camera_y
            if 0 <= screen_x < SCREEN_WIDTH and 0 <= screen_y < SCREEN_HEIGHT:
                pygame.draw.rect(screen, GREEN, pygame.Rect(screen_x, screen_y, GRID_SIZE, GRID_SIZE))
