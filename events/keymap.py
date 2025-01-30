from constants.options import Options, KeyMapEntry
from utils.list import LimitedList

class Keymap:
    def __init__(self):
        self.history = LimitedList[KeyMapEntry](100)
        self.options = Options()

        self.actions = {
            "toggle_debug": self.options.set_key("debug", not self.options.get_key("debug")),
            "toggle_menu": self.options.set_key("menu", not self.options.get_key("menu"))
        }

    def handle_key(self, key: str):
        keymap_entry = self.options.get_keymap_entry(key)
        if keymap_entry is None:
            return

        action = keymap_entry["action"]

        self.history.append(keymap_entry)

        ac = self.actions[action]
        if ac is not None:
            ac()
