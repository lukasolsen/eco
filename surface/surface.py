from pygame.surface import Surface
import pygame
from utils.singleton import Singleton

class SurfaceEditor(metaclass=Singleton):
    def __init__(self, screen: Surface):
        self.screen = screen

    def draw_rect(self, color: tuple[int, int, int], x: int, y: int, width: int, height: int):
        pygame.draw.rect(self.screen, color, (x, y, width, height))

    def draw_text(self, text: str, color: tuple[int, int, int], x: int, y: int, font: pygame.font.Font):
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))

    def draw_surface(self, surface: Surface, x: int, y: int):
        self.screen.blit(surface, (x, y))
