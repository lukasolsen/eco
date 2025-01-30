from typing import TypeVar, cast, List
from utils.key import KeyMapEntry
from utils.singleton import Singleton

WIDTH, HEIGHT = 800, 600
GRID_SIZE = 20  # Size of each grid cell
CHUNK_SIZE = 16  # Number of grids per chunk

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

T = TypeVar("T", bool, int, float, str)

class Options(metaclass=Singleton):
    def __init__(self):
        self.options: dict[str, bool | int | float | str] = {
            "debug": False,
            "menu": False,
            "command": False
        }

        self.blacklisted_keys: list[str] = [
            "up", "right", "down", "left"
        ]

        self.keymap: List[KeyMapEntry] = [
            {
                "key": "f3",
                "action": "toggle_debug",
                "description": "Toggle debug mode"
            },
            {
                "key": "return",
                "action": "toggle_menu",
                "description": "Toggle the menu"
            }
        ]

    def get_key(self, key: str) -> T:
        return cast(T, self.options[key])  # Explicitly cast the return type

    def set_key(self, key: str, value: T):
        self.options[key] = value

    def get_keymap_entry(self, key: str) -> KeyMapEntry | None:
        for entry in self.keymap:
            if entry["key"] == key:
                return entry

        if self.get_key("debug"):
            if key not in self.blacklisted_keys:
                print(f"Key {key} not found in keymap")

        return None

    def get_options(self):
        return self.options

    def get_keymap(self):
        return self.keymap
