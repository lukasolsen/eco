import os
import json
import platform
from datetime import datetime

class GameDataManager:
    def __init__(self, game_name="WorldSimulation"):
        self.game_name = game_name
        self.game_data_dir = self._get_game_data_dir()
        self.crash_reports_dir = os.path.join(self.game_data_dir, "crash_reports")
        self.logs_dir = os.path.join(self.game_data_dir, "logs")
        self.metadata_file = os.path.join(self.game_data_dir, "metadata.json")

        # Ensure directories exist
        self._create_directories()

    def _get_game_data_dir(self):
        """Get the appropriate game data directory based on the operating system."""
        system = platform.system()

        if system == "Windows":
            app_data = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA") or "."
            return os.path.join(app_data, self.game_name)
        elif system == "Darwin":
            return os.path.join(os.path.expanduser("~/Library/Application Support"), self.game_name)
        elif system == "Linux":
            return os.path.join(os.path.expanduser("~/.local/share"), self.game_name)
        else:
            raise OSError(f"Unsupported operating system: {system}")

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.game_data_dir, exist_ok=True)
        os.makedirs(self.crash_reports_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def save_crash_report(self, error_message, traceback):
        """Save a crash report to the crash_reports directory."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        crash_report_file = os.path.join(self.crash_reports_dir, f"crash_{timestamp}.txt")

        with open(crash_report_file, "w") as f:
            f.write(f"Crash Report ({timestamp}):\n")
            f.write(f"Error: {error_message}\n")
            f.write("Traceback:\n")
            f.write(traceback)

    def save_log(self, log_message):
        """Save a log message to the logs directory."""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self.logs_dir, f"log_{timestamp}.txt")

        with open(log_file, "a") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}\n")

    def load_metadata(self):
        """Load game metadata (e.g., last played time, has played before)."""
        if not os.path.exists(self.metadata_file):
            return {"has_played_before": False, "last_played": None}

        with open(self.metadata_file, "r") as f:
            return json.load(f)

    def save_metadata(self, has_played_before=True, last_played=None):
        """Save game metadata."""
        metadata = {
            "has_played_before": has_played_before,
            "last_played": last_played or datetime.now().isoformat(),
        }

        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)
