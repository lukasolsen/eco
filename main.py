import traceback
import sys
from game import Game
import pygame
from utils.data import GameDataManager

def main():
    try:
        data_manager = GameDataManager()
    except Exception:
        sys.exit(1)

    try:
        Game(data_manager).run()
    except Exception as e:
        data_manager.save_crash_report(str(e), traceback.format_exc())
        pygame.quit()
        sys.exit(1)

if __name__ == "__main__":
    main()
