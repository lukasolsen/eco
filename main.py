import pygame
from classes.camera import Camera
from classes.world import World
from constants.options import WIDTH, HEIGHT, CHUNK_SIZE, GRID_SIZE, WHITE, BLACK, Options
from events.event import EventManager

class Game:
    def __init__(self):
        self.__setup_pygame()

        self.camera = Camera(WIDTH, HEIGHT)
        self.world = World(CHUNK_SIZE)
        self.event_manager = EventManager()
        self.options = Options()

    def __setup_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("World Simulation")
        self.clock = pygame.time.Clock()

    def draw_world(self):
        visible_chunks = self.world.get_visible_chunks(*self.camera.get_offset(), WIDTH, HEIGHT)
        self.world.update_loaded_chunks(visible_chunks)

        for chunk_x, chunk_y in visible_chunks:
            chunk = self.world.load_chunk(chunk_x, chunk_y)
            chunk_data = chunk.get_data()

            for row in range(self.world.chunk_size):
                for col in range(self.world.chunk_size):
                    block_type = chunk_data[row][col]
                    world_x = (chunk_x * self.world.chunk_size + col) * GRID_SIZE
                    world_y = (chunk_y * self.world.chunk_size + row) * GRID_SIZE

                    # Apply camera offset and zoom
                    screen_x, screen_y = self.camera.apply_offset(world_x, world_y)
                    screen_x *= self.camera.get_zoom_level()
                    screen_y *= self.camera.get_zoom_level()
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

                    biome_codes = {
                        "water": "W",
                        "river_water": "R",
                        "grass": "G",
                        "sand": "S",
                        "rock": "C",
                        "snow": "N",
                        "river": "r",
                    }

                    # Elevation-based shading (lighter/darker variations)
                    color = biome_colors.get(block_type, (255, 255, 255))
                    if block_type in ["grass", "sand", "rock"]:
                        elevation = row / self.world.chunk_size  # Normalize elevation
                        brightness = int(30 * elevation)  # Adjust brightness
                        color = (max(0, color[0] - brightness),
                                max(0, color[1] - brightness),
                                max(0, color[2] - brightness))

                    # Draw the block
                    if 0 <= screen_x < WIDTH and 0 <= screen_y < HEIGHT:
                        pygame.draw.rect(self.screen, color, (screen_x, screen_y, block_size, block_size))
                        pygame.draw.rect(self.screen, BLACK, (screen_x, screen_y, block_size, block_size), 1)

                        if self.options.get_key("debug"):
                            font = pygame.font.Font(None, 20)
                            text = font.render(str(biome_codes.get(block_type, " ")), True, BLACK)
                            self.screen.blit(text, (screen_x + 5, screen_y + 5))

        # Debug info: Show chunk borders and coordinates
        if self.options.get_key("debug"):
            for chunk_x, chunk_y in visible_chunks:
                chunk_screen_x, chunk_screen_y = self.camera.apply_offset(
                    chunk_x * GRID_SIZE * self.world.chunk_size,
                    chunk_y * GRID_SIZE * self.world.chunk_size
                )
                chunk_screen_x *= self.camera.get_zoom_level()
                chunk_screen_y *= self.camera.get_zoom_level()
                chunk_size = GRID_SIZE * self.world.chunk_size * self.camera.get_zoom_level()

                pygame.draw.rect(self.screen, BLACK, (chunk_screen_x, chunk_screen_y, chunk_size, chunk_size), 1)

                font = pygame.font.Font(None, 20)
                text = font.render(f"{chunk_x}, {chunk_y}", True, BLACK)
                self.screen.blit(text, (chunk_screen_x + 5, chunk_screen_y + 5))

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.camera.handle_event(event)

                self.event_manager.perform_event(event)

            self.screen.fill(WHITE)
            self.draw_world()
            pygame.display.flip()
            self.clock.tick(120)

if __name__ == "__main__":
    Game().run()
