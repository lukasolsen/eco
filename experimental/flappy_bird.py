import pygame
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Game Constants
WIDTH, HEIGHT = 500, 500
GROUND_HEIGHT = 100
BIRD_SIZE = 24
PIPE_WIDTH, PIPE_GAP = 50, 150
FPS = 120

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (200, 0, 0)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird AI")
clock = pygame.time.Clock()

# Bird and Pipe Classes
class Bird:
    def __init__(self):
        self.x = 50
        self.y = HEIGHT // 2
        self.velocity = 0
        self.gravity = 0.5
        self.flap_power = -8
        self.alive = True

    def flap(self):
        self.velocity = self.flap_power

    def update(self):
        self.velocity += self.gravity
        self.y += self.velocity
        if self.y >= HEIGHT - GROUND_HEIGHT - BIRD_SIZE or self.y <= 0:
            self.alive = False

    def render(self):
        pygame.draw.rect(screen, RED, (self.x, self.y, BIRD_SIZE, BIRD_SIZE))


class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = random.randint(50, HEIGHT - PIPE_GAP - GROUND_HEIGHT)
        self.passed = False

    def update(self):
        self.x -= 5

    def render(self):
        pygame.draw.rect(screen, GREEN, (self.x, 0, PIPE_WIDTH, self.height))
        pygame.draw.rect(screen, GREEN, (self.x, self.height + PIPE_GAP, PIPE_WIDTH, HEIGHT - GROUND_HEIGHT - self.height - PIPE_GAP))


# DQAgent for Flappy Bird
class DQAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.01
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


# Main Game Loop with Training
def main():
    episodes = 500
    agent = DQAgent(state_size=4, action_size=2)
    global_score = 0

    for episode in range(episodes):
        bird = Bird()
        pipes = [Pipe(WIDTH + i * 200) for i in range(2)]
        score = 0
        state = np.array([bird.y, bird.velocity, pipes[0].x, pipes[0].height]) / HEIGHT
        state = np.reshape(state, [1, 4])

        while bird.alive:
            action = agent.act(state)
            if action == 1:
                bird.flap()

            bird.update()
            for pipe in pipes:
                pipe.update()
                if pipe.x + PIPE_WIDTH < 0:
                    pipes.remove(pipe)
                    pipes.append(Pipe(WIDTH))
                if not pipe.passed and bird.x > pipe.x + PIPE_WIDTH:
                    pipe.passed = True
                    score += 1
                    global_score = max(score, global_score)
                if pipe.x < bird.x < pipe.x + PIPE_WIDTH:
                    if bird.y < pipe.height or bird.y > pipe.height + PIPE_GAP:
                        bird.alive = False

            next_state = np.array([bird.y, bird.velocity, pipes[0].x, pipes[0].height]) / HEIGHT
            next_state = np.reshape(next_state, [1, 4])
            reward = 1 if bird.alive else -100
            agent.remember(state, action, reward, next_state, not bird.alive)
            state = next_state

            screen.fill(WHITE)
            bird.render()
            for pipe in pipes:
                pipe.render()
            pygame.display.flip()
            clock.tick(FPS)

        agent.replay()
        print(f"Episode {episode + 1}/{episodes}, Score: {score}")

    pygame.quit()


if __name__ == "__main__":
    pygame.init()
    try:
        main()
    finally:
        pygame.quit()
