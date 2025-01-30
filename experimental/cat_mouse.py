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
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Initialize Pygame
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cat and Mouse")
clock = pygame.time.Clock()

# Cat and Mouse Agent Classes
class DQAgent:
    def __init__(self, state_size, action_size, name):
        self.state_size = state_size
        self.action_size = action_size
        self.name = name
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = 128
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
def render_text(surface, text, position, font_size=20, color=BLACK):
    """Utility function to render text on the screen."""
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, position)

class Game:
    def __init__(self):
        self.cat = [5, 5]
        self.mouse = [15, 15]
        self.cat_agent = DQAgent(4, 4, "Cat")
        self.mouse_agent = DQAgent(4, 4, "Mouse")
        self.episode_count = 0
        self.reset()

    def reset(self):
        self.cat = [5, 5]
        self.mouse = [15, 15]
        self.episode_count += 1
        return self.get_state()

    def get_state(self):
        return np.array([self.cat[0], self.cat[1], self.mouse[0], self.mouse[1]]) / max(NUM_ROWS, NUM_COLS)

    def step(self, action_cat, action_mouse):
        # Action mapping: 0 = up, 1 = down, 2 = left, 3 = right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Update positions
        self.cat[0] += moves[action_cat][0]
        self.cat[1] += moves[action_cat][1]
        self.mouse[0] += moves[action_mouse][0]
        self.mouse[1] += moves[action_mouse][1]

        # Keep within bounds
        self.cat[0] = max(0, min(NUM_ROWS - 1, self.cat[0]))
        self.cat[1] = max(0, min(NUM_COLS - 1, self.cat[1]))
        self.mouse[0] = max(0, min(NUM_ROWS - 1, self.mouse[0]))
        self.mouse[1] = max(0, min(NUM_COLS - 1, self.mouse[1]))

        # Check for end condition
        done = self.cat == self.mouse
        reward_cat = 1 if done else -0.01
        reward_mouse = -1 if done else 0.01

        next_state = self.get_state()
        return next_state, reward_cat, reward_mouse, done

    def render(self, elapsed_time, total_reward_cat, total_reward_mouse):
        # Clear screen
        screen.fill(WHITE)

        # Draw grid
        pygame.draw.rect(screen, RED, pygame.Rect(self.cat[1] * GRID_SIZE, self.cat[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(screen, BLUE, pygame.Rect(self.mouse[1] * GRID_SIZE, self.mouse[0] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # Render text
        render_text(screen, "Cat and Mouse Game", (10, 10), font_size=24, color=BLACK)
        render_text(screen, f"Time: {elapsed_time // 1000}s", (10, 40))
        render_text(screen, f"Cat Reward: {total_reward_cat:.2f}", (10, 70))
        render_text(screen, f"Mouse Reward: {total_reward_mouse:.2f}", (10, 100))
        render_text(screen, f"Episode: {self.episode_count}", (10, 130))

        # Display agent details
        render_text(screen, "Cat Agent", (self.cat[1] * GRID_SIZE + 5, self.cat[0] * GRID_SIZE - 15), font_size=16, color=RED)
        render_text(screen, f"Epsilon: {self.cat_agent.epsilon:.2f}", (10, 160))
        render_text(screen, "Mouse Agent", (self.mouse[1] * GRID_SIZE + 5, self.mouse[0] * GRID_SIZE - 15), font_size=16, color=BLUE)

        pygame.display.flip()


def main():
    game = Game()
    episodes = 1000
    max_time = 60 * 1000  # 1 minute in milliseconds

    for episode in range(episodes):
        state = game.reset()
        state = np.reshape(state, [1, 4])
        done = False
        total_reward_cat = 0
        total_reward_mouse = 0
        start_time = pygame.time.get_ticks()  # Track start time

        while not done:
            current_time = pygame.time.get_ticks()  # Check current time
            elapsed_time = current_time - start_time

            if elapsed_time >= max_time:
                print("Time's up! Mouse wins!")
                break  # End the game if time limit is reached

            action_cat = game.cat_agent.act(state)
            action_mouse = game.mouse_agent.act(state)

            next_state, reward_cat, reward_mouse, done = game.step(action_cat, action_mouse)
            next_state = np.reshape(next_state, [1, 4])

            game.cat_agent.remember(state, action_cat, reward_cat, next_state, done)
            game.mouse_agent.remember(state, action_mouse, reward_mouse, next_state, done)

            state = next_state
            total_reward_cat += reward_cat
            total_reward_mouse += reward_mouse

            if episode % 100 == 0:  # Render every 100 episodes
                game.render(elapsed_time, total_reward_cat, total_reward_mouse)
                clock.tick(10)  # Limit to 10 frames per second
            else:
                game.render(elapsed_time, total_reward_cat, total_reward_mouse)
                clock.tick()

        if elapsed_time < max_time and done:  # If cat catches mouse before time runs out
            print("Cat caught the mouse!")
        elif elapsed_time >= max_time and not done:  # If time runs out
            total_reward_mouse += 1  # Reward the mouse for surviving

        game.cat_agent.replay()
        game.mouse_agent.replay()

        print(f"Episode {episode + 1}/{episodes}: Cat Reward = {total_reward_cat:.2f}, Mouse Reward = {total_reward_mouse:.2f}")

    pygame.quit()


if __name__ == "__main__":
    pygame.init()
    try:
        main()
    finally:
        pygame.quit()
