import noise
import random
import typing
from classes.chunk import Chunk
from classes.biome import BiomeGenerator

class World:
    def __init__(self, chunk_size: int = 16, grid_size: int = 20):
        self.seed = 1
        self.chunk_size = chunk_size
        self.grid_size = grid_size
        self.loaded_chunks: typing.Dict[typing.Tuple[int, int], Chunk] = {}

        self.biome_generator = BiomeGenerator(seed=self.seed)

        # Noise parameters
        self.base_scale = 100.0
        self.detail_scale = 30.0
        self.mountain_scale = 200.0
        self.biome_scale = 150.0
        self.domain_warp_scale = 50.0

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
        """Generate terrain using layered noise and domain warping."""
        chunk_data = [["empty" for _ in range(self.chunk_size)] for _ in range(self.chunk_size)]

        for row in range(self.chunk_size):
            for col in range(self.chunk_size):
                world_x = chunk_x * self.chunk_size + col
                world_y = chunk_y * self.chunk_size + row

                # Domain warping for natural terrain
                warp_x, warp_y = self.domain_warp(world_x, world_y, self.domain_warp_scale)

                # Base elevation (hills and valleys)
                base_elevation = self.noise(warp_x, warp_y, self.base_scale, octaves=4)

                # Detail noise (small bumps and roughness)
                detail_elevation = self.noise(warp_x, warp_y, self.detail_scale, octaves=2)

                # Mountain noise (large, dramatic mountains)
                mountain_elevation = self.noise(warp_x, warp_y, self.mountain_scale, octaves=6)

                # Combine elevation layers
                elevation = base_elevation + 0.5 * detail_elevation + 0.2 * mountain_elevation

                # Biome noise (determines biome type)
                biome_value = self.noise(world_x, world_y, self.biome_scale, octaves=1)

                # Assign biome and block type
                chunk_data[row][col] = self.get_block_type(elevation, biome_value)

        # Generate rivers more naturally
        chunk_data = self.generate_rivers(chunk_x, chunk_y, chunk_data)

        return chunk_data

    def noise(self, x: float, y: float, scale: float, octaves: int = 1) -> float:
        """Generate Perlin noise at a given coordinate."""
        return noise.pnoise2(x / scale, y / scale, octaves=octaves, base=self.seed)

    def domain_warp(self, x: float, y: float, scale: float) -> typing.Tuple[float, float]:
        """Apply domain warping to create natural-looking terrain."""
        warp_x = self.noise(x, y, scale)
        warp_y = self.noise(y, x, scale)
        return x + warp_x * 20, y + warp_y * 20

    def get_block_type(self, elevation: float, biome_value: float) -> str:
        """Determine block type based on elevation and biome."""
        if elevation < 0.3:
            return "water"
        elif elevation < 0.6:
            if biome_value < -0.5:
                return "sand"  # Desert
            elif biome_value < 0.0:
                return "grass"  # Plains
            elif biome_value < 0.5:
                return "grass"  # Forest
            else:
                return "snow"  # Snowy biome
        else:
            return "rock"  # Mountains

    def generate_rivers(self, chunk_x: int, chunk_y: int, chunk_data):
        """Uses noise-based river generation instead of random placement."""
        river_noise_scale = 50  # Scale for river noise
        river_threshold = 0.1  # Threshold for river generation

        for row in range(self.chunk_size):
            for col in range(self.chunk_size):
                world_x = chunk_x * self.chunk_size + col
                world_y = chunk_y * self.chunk_size + row

                # Generate river noise
                nx = world_x / river_noise_scale
                ny = world_y / river_noise_scale
                river_noise = noise.pnoise2(nx, ny, octaves=2, persistence=0.5, lacunarity=2.0, repeatx=2048, repeaty=2048, base=self.seed)

                # Rivers form in valleys and follow the terrain
                if 0.3 < self.noise(world_x, world_y, self.base_scale) < 0.5 and abs(river_noise) < river_threshold:
                    chunk_data[row][col] = "river_water"

        return chunk_data

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
