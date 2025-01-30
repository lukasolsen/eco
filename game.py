import pygame
from classes.camera import Camera
from classes.world import World
from constants.options import CHUNK_SIZE, GRID_SIZE, BLACK, Options
from events.event import EventManager
from utils.data import GameDataManager

class Game:
    def __init__(self, data_manager: GameDataManager):
        self.__setup_pygame()

        self.camera = Camera()
        self.world = World(CHUNK_SIZE)
        self.event_manager = EventManager()
        self.options = Options()

        # Cache for rendered chunks
        self.chunk_cache = {}

    def __setup_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)
        pygame.display.set_caption("World Simulation")
        self.clock = pygame.time.Clock()

    def draw_world(self):
        screen_width, screen_height = pygame.display.get_window_size()
        visible_chunks = self.world.get_visible_chunks(
            *self.camera.get_offset(), screen_width, screen_height, self.camera.get_zoom_level()
        )
        self.world.update_loaded_chunks(visible_chunks)

        for chunk_x, chunk_y in visible_chunks:
            # Check if the chunk is already cached
            cache_key = (chunk_x, chunk_y, self.camera.get_zoom_level())
            if cache_key in self.chunk_cache:
                # Draw the cached chunk
                chunk_surface = self.chunk_cache[cache_key]
                screen_x, screen_y = self.camera.apply_offset(
                    chunk_x * self.world.chunk_size * GRID_SIZE,
                    chunk_y * self.world.chunk_size * GRID_SIZE
                )
                screen_x *= self.camera.get_zoom_level()
                screen_y *= self.camera.get_zoom_level()
                self.screen.blit(chunk_surface, (screen_x, screen_y))
            else:
                # Render the chunk and cache it
                chunk_surface = self._render_chunk(chunk_x, chunk_y)
                self.chunk_cache[cache_key] = chunk_surface
                screen_x, screen_y = self.camera.apply_offset(
                    chunk_x * self.world.chunk_size * GRID_SIZE,
                    chunk_y * self.world.chunk_size * GRID_SIZE
                )
                screen_x *= self.camera.get_zoom_level()
                screen_y *= self.camera.get_zoom_level()
                self.screen.blit(chunk_surface, (screen_x, screen_y))

    def _render_chunk(self, chunk_x, chunk_y):
        """Render a single chunk to a Pygame surface and return it."""
        chunk = self.world.load_chunk(chunk_x, chunk_y)
        chunk_data = chunk.get_data()

        # Create a surface for the chunk
        chunk_size_pixels = self.world.chunk_size * GRID_SIZE * self.camera.get_zoom_level()
        chunk_surface = pygame.Surface((chunk_size_pixels, chunk_size_pixels), pygame.SRCALPHA)

        for row in range(self.world.chunk_size):
            for col in range(self.world.chunk_size):
                block_type = chunk_data[row][col]
                world_x = col * GRID_SIZE
                world_y = row * GRID_SIZE

                # Apply zoom level
                block_size = GRID_SIZE * self.camera.get_zoom_level()

                # Define biome-based colors
                biome_colors = {
                    "water": (28, 107, 160),  # Deep water
                    "river_water": (20, 90, 200),  # River water
                    "grass": (34, 139, 34),   # Grasslands
                    "sand": (237, 201, 175),  # Desert
                    "rock": (139, 69, 19),    # Mountains
                    "snow": (240, 240, 240),  # Snowy biomes
                    "river": (20, 90, 200),   # River color
                }

                # Elevation-based shading
                color = biome_colors.get(block_type, (255, 255, 255))
                if block_type in ["grass", "sand", "rock"]:
                    elevation = row / self.world.chunk_size  # Normalize elevation
                    brightness = int(30 * elevation)  # Adjust brightness
                    color = (max(0, color[0] - brightness),
                             max(0, color[1] - brightness),
                             max(0, color[2] - brightness))

                # Draw the block on the chunk surface
                pygame.draw.rect(chunk_surface, color, (world_x, world_y, block_size, block_size))

        return chunk_surface

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    self.chunk_cache = {}  # Invalidate cache on window resize
                elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_z:  # Example: Zoom in/out
                        self.chunk_cache = {}  # Invalidate cache on zoom change
                self.camera.handle_event(event)
                self.event_manager.perform_event(event)

            self.screen.fill((0, 0, 0))
            self.draw_world()

            if self.options.get_key("debug"):
                self._draw_debug_info()

            pygame.display.flip()
            self.clock.tick(120)

    def _draw_debug_info(self):
        """Draw debug information on the screen."""
        screen_width, screen_height = pygame.display.get_surface().get_size()
        font = pygame.font.Font(None, 20)

        data = [
            f"Camera: {self.camera.get_offset()}",
            f"Screen: {screen_width}x{screen_height}",
            f"Zoom: {self.camera.get_zoom_level()}",
            f"Chunks: {len(self.world.loaded_chunks)}",
            f"FPS: {int(self.clock.get_fps())}",
            f"Seed: {self.world.seed}",
        ]

        for i, text in enumerate(data):
            text = font.render(text, True, BLACK)
            self.screen.blit(text, (5, 5 + 20 * i))

        text = font.render(f"Version: {self.options.get_key('version')}", True, BLACK)
        self.screen.blit(text, (5, screen_height - 25))
