import pygame
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random

# Game Constants
WIDTH, HEIGHT = 500, 500
GRID_SIZE = 20
NUM_ROWS, NUM_COLS = HEIGHT // GRID_SIZE, WIDTH // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Initialize Pygame
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cell Simulation")
clock = pygame.time.Clock()

class DQAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            layers.Dense(128, activation="relu", input_dim=self.state_size),
            layers.Dense(128, activation="relu"),
            layers.Dense(self.action_size, activation="linear")
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            self.memory.pop(0)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class Cell:
    def __init__(self, x, y, size, agent):
        self.x = x
        self.y = y
        self.size = size
        self.energy = 100
        self.agent = agent

    def get_state(self, food_positions, other_cells):
        food_distances = [((fx - self.x)**2 + (fy - self.y)**2)**0.5 for fx, fy in food_positions]
        nearest_food = min(food_distances) if food_distances else 1.0
        return np.array([self.x / NUM_ROWS, self.y / NUM_COLS, self.energy / 100, nearest_food])

    def move(self, action):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if action < len(moves):
            dx, dy = moves[action]
            self.x = max(0, min(NUM_ROWS - 1, self.x + dx))
            self.y = max(0, min(NUM_COLS - 1, self.y + dy))

class Game:
    def __init__(self):
        self.cells = [Cell(random.randint(0, NUM_ROWS - 1), random.randint(0, NUM_COLS - 1), random.randint(1, 5), DQAgent(4, 4)) for _ in range(10)]
        self.food = [(random.randint(0, NUM_ROWS - 1), random.randint(0, NUM_COLS - 1)) for _ in range(20)]

    def step(self):
        new_food = [(random.randint(0, NUM_ROWS - 1), random.randint(0, NUM_COLS - 1)) for _ in range(5)]
        self.food.extend(new_food)

        for cell in self.cells:
            state = cell.get_state(self.food, self.cells)
            state = np.reshape(state, [1, 4])
            action = cell.agent.act(state)
            cell.move(action)

            # Check for food consumption
            for food in self.food:
                if (cell.x, cell.y) == food:
                    cell.energy += 20
                    self.food.remove(food)

            # Check for reproduction
            if cell.energy >= 200:
                cell.energy -= 100
                self.cells.append(Cell(cell.x, cell.y, cell.size, DQAgent(4, 4)))

            # Decay energy
            cell.energy -= 1

        # Remove dead cells
        self.cells = [cell for cell in self.cells if cell.energy > 0]

    def render(self):
        screen.fill(WHITE)

        # Draw food as green blocks representing "nutrients"
        for food in self.food:
            pygame.draw.rect(screen, GREEN, pygame.Rect(food[1] * GRID_SIZE, food[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # Draw cells as red circles representing "organisms"
        for cell in self.cells:
            pygame.draw.circle(screen, RED, (cell.y * GRID_SIZE + GRID_SIZE // 2, cell.x * GRID_SIZE + GRID_SIZE // 2), cell.size)

        # Add a legend
        font = pygame.font.Font(None, 24)
        legend_food = font.render("Green Blocks: Nutrients", True, BLACK)
        legend_cells = font.render("Red Circles: Organisms", True, BLACK)
        screen.blit(legend_food, (10, HEIGHT - 40))
        screen.blit(legend_cells, (10, HEIGHT - 20))

        pygame.display.flip()


def main():
    game = Game()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        game.step()
        game.render()
        clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    pygame.init()
    main()
