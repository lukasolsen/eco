from pygame.event import Event
from utils.list import LimitedList
import pygame

from events.keymap import Keymap

class EventManager:
    def __init__(self):
        self.events = LimitedList[Event](100)

        self._keymap = Keymap()

    def perform_event(self, event: Event):
        self.events.append(event)

        if event.type == pygame.KEYDOWN:
            self._keymap.handle_key(pygame.key.name(event.key))
