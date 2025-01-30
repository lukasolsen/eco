import noise
import random
import typing
from classes.chunk import Chunk, ChunkType

class World:
    def __init__(self, chunk_size: int = 16, grid_size: int = 20):
        self.seed = 0  # World seed for procedural generation
        self.chunk_size = chunk_size
        self.grid_size = grid_size
        self.loaded_chunks: typing.Dict[typing.Tuple[int, int], Chunk] = {}

    def generate_world(self, seed: int):
        """Sets the seed for world generation and initializes the world."""
        self.seed = seed
        random.seed(seed)

    def load_chunk(self, chunk_x: int, chunk_y: int) -> Chunk:
        """Load or retrieve a chunk from the cache."""
        chunk_key = (chunk_x, chunk_y)
        if chunk_key not in self.loaded_chunks:
            chunk = Chunk(chunk_x, chunk_y, self.chunk_size)
            self.loaded_chunks[chunk_key] = chunk
            self.populate_chunk(chunk)
        return self.loaded_chunks[chunk_key]

    def unload_chunk(self, chunk_x: int, chunk_y: int):
        """Removes a chunk from the loaded chunks to free up memory."""
        chunk_key = (chunk_x, chunk_y)
        if chunk_key in self.loaded_chunks:
            del self.loaded_chunks[chunk_key]

    def generate_chunk_data(self, chunk_x: int, chunk_y: int) -> typing.List[typing.List[str]]:
        """Generates data for a chunk based on its position."""
        chunk_data = self.create_terrain(chunk_x, chunk_y)
        return chunk_data

    def create_terrain(self, chunk_x: int, chunk_y: int) -> typing.List[typing.List[str]]:
        """Generate terrain using Perlin noise and additional features like rocks and water."""
        chunk_data = [["empty" for _ in range(self.chunk_size)] for _ in range(self.chunk_size)]

        # Define elevation parameters
        elevation_map = self.generate_perlin_noise(chunk_x, chunk_y)

        # Place terrain features based on elevation
        for row in range(self.chunk_size):
            for col in range(self.chunk_size):
                elevation = elevation_map[row][col]

                # Water: low elevation
                if elevation < 0.3:
                    chunk_data[row][col] = "water"
                # Grass: medium elevation
                elif elevation < 0.6:
                    chunk_data[row][col] = "grass"
                # Rocks: high elevation (mountains)
                elif elevation >= 0.6:
                    chunk_data[row][col] = "rock"

        return chunk_data

    def generate_perlin_noise(self, chunk_x: int, chunk_y: int) -> typing.List[typing.List[float]]:
        """Generate a 2D Perlin noise map for terrain elevation."""
        elevation_map = []
        scale = 100  # This affects the "zoom" level of the terrain features
        for row in range(self.chunk_size):
            noise_row = []
            for col in range(self.chunk_size):
                # Generate Perlin noise at each position
                nx = (chunk_x * self.chunk_size + col) / scale
                ny = (chunk_y * self.chunk_size + row) / scale

                # Normalize the seed value to prevent overflow
                seed = self.seed + chunk_x * 31 + chunk_y * 37
                base = (seed % 1024) + 1  # Ensure it's never zero

                elevation = noise.pnoise2(nx, ny, octaves=4, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, base=base)
                noise_row.append(elevation)
            elevation_map.append(noise_row)
        return elevation_map

    def get_visible_chunks(self, offset_x: float, offset_y: float, screen_width: int, screen_height: int):
        """Calculate the visible chunks based on the camera offset and screen size."""
        start_x = int(offset_x / self.grid_size / self.chunk_size)
        start_y = int(offset_y / self.grid_size / self.chunk_size)
        end_x = int((offset_x + screen_width) / self.grid_size / self.chunk_size)
        end_y = int((offset_y + screen_height) / self.grid_size / self.chunk_size)

        visible_chunks = {(x, y) for x in range(start_x - 1, end_x + 2) for y in range(start_y - 1, end_y + 2)}
        return visible_chunks

    def get_all_chunks(self):
        """Return all loaded chunks."""
        return self.loaded_chunks.values()

    def update_loaded_chunks(self, visible_chunks: typing.Set[typing.Tuple[int, int]]):
        """Loads visible chunks and unloads chunks no longer visible."""
        current_keys = set(self.loaded_chunks.keys())

        # Unload chunks no longer visible
        for chunk_pos in current_keys - visible_chunks:
            self.unload_chunk(*chunk_pos)

        # Load new chunks that are now visible
        for chunk_x, chunk_y in visible_chunks:
            if (chunk_x, chunk_y) not in self.loaded_chunks:
                self.load_chunk(chunk_x, chunk_y)


    def populate_chunk(self, chunk: Chunk):
        """Populate the chunk with terrain data."""
        chunk.chunk_data = self.create_terrain(chunk.chunk_x, chunk.chunk_y)
