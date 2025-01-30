from scipy.spatial import Voronoi
import numpy as np
import random

class BiomeGenerator:
    def __init__(self, seed: int, num_biomes: int = 10):
        self.seed = seed
        random.seed(seed)
        self.biome_centers = [(random.randint(-1000, 1000), random.randint(-1000, 1000)) for _ in range(num_biomes)]
        self.voronoi = Voronoi(self.biome_centers)
        self.biome_types = ["plains", "forest", "desert", "snow", "swamp", "mountain"]
        self.biome_weights = [0.5, 0.2, 0.1, 0.1, 0.05, 0.05]  # Weighted probabilities for biomes

    def get_biome(self, x: int, y: int) -> str:
        """Determine biome type based on Voronoi regions and weighted probabilities."""
        point = np.array([x, y])
        distances = np.linalg.norm(self.voronoi.points - point, axis=1)
        closest = np.argmin(distances)
        # Use weighted random selection to favor plains
        return random.choices(self.biome_types, weights=self.biome_weights, k=1)[0]
