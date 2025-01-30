import pygame

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
GRID_SIZE = 20
WORLD_ROWS, WORLD_COLS = 100, 100
FPS = 30
INITIAL_POPULATION = 20
MAX_POPULATION = 500
CAMERA_SPEED = 10

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Initialize Pygame
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Civilization Simulation with RL")
clock = pygame.time.Clock()
