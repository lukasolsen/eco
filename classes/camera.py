import pygame
from pygame.event import Event

class Camera:
    def __init__(self, width, height):
        self.offset_x = 0
        self.offset_y = 0
        self.zoom_level = 1.0
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        self.width = width
        self.height = height

    def handle_event(self, event: Event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                self.dragging = True
                self.last_mouse_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                dx, dy = event.rel
                self.offset_x -= dx
                self.offset_y -= dy
        elif event.type == pygame.MOUSEWHEEL:
            if event.y > 0:  # Scroll up
                self.zoom_level += 0.1
            elif event.y < 0:  # Scroll down
                self.zoom_level -= 0.1
            self.zoom_level = max(0.1, self.zoom_level)  # Limit zoom level
        elif event.type == pygame.KEYDOWN:
            self._handle_keydown(event)

    def _handle_keydown(self, event: Event):
        if event.key == pygame.K_LEFT:
            self.offset_x -= 10
        elif event.key == pygame.K_RIGHT:
            self.offset_x += 10
        elif event.key == pygame.K_UP:
            self.offset_y -= 10
        elif event.key == pygame.K_DOWN:
            self.offset_y += 10

    def apply_offset(self, x, y):
        return x - self.offset_x, y - self.offset_y

    def get_offset(self):
        return self.offset_x, self.offset_y

    def get_zoom_level(self):
        return self.zoom_level
