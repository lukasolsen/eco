import random
import numpy as np
import pygame
from utils.constants import *
from utils.ai import create_model, mutate_model

class Agent:
    def __init__(self, x, y, gender, generation):
        self.x = x
        self.y = y
        self.gender = gender
        self.generation = generation
        self.energy = random.randint(50, 100)
        self.alive = True
        self.age = 0
        self.model = create_model(input_dim=6, output_dim=4)  # State: [x, y, energy, food_dist, agents_nearby]
        self.memory = []  # For experience replay
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def move(self, action):
        dx, dy = [(0, 1), (1, 0), (0, -1), (-1, 0)][action]
        self.x = (self.x + dx) % WORLD_COLS
        self.y = (self.y + dy) % WORLD_ROWS
        self.energy -= 1

    def get_state(self, food, agents):
        food_dist = min((((fx - self.x) ** 2 + (fy - self.y) ** 2) ** 0.5 for fx, fy in food), default=0)
        agents_nearby = sum(1 for a in agents if (a.x - self.x) ** 2 + (a.y - self.y) ** 2 <= 4)
        return np.array([self.x / WORLD_COLS, self.y / WORLD_ROWS, self.energy / 100, food_dist, agents_nearby, self.age / 120])

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, 3)  # Explore
        q_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values)  # Exploit

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.model.predict(next_state[np.newaxis], verbose=0))
            target_f = self.model.predict(state[np.newaxis], verbose=0)
            target_f[0][action] = target
            self.model.fit(state[np.newaxis], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def seek_food(self, food_in_sight):
      nearest_food = min(food_in_sight, key=lambda f: (f[0] - self.x) ** 2 + (f[1] - self.y) ** 2)
      dx = np.sign(nearest_food[0] - self.x)
      dy = np.sign(nearest_food[1] - self.y)
      self.x = (self.x + dx) % WORLD_COLS
      self.y = (self.y + dy) % WORLD_ROWS
      self.energy -= 1

    def seek_mate(self, nearby_agents, agents, generation):
      mate = next((a for a in nearby_agents if a.gender != self.gender and a.energy > 50), None)
      if mate:
          self.reproduce(agents, generation)
      else:
          # Random movement if no mate found
          self.move(random.randint(0, 3))

    def aggressive_or_flee(self, nearby_agents):
      if len(nearby_agents) > 2 and self.energy < 40:  # Outnumbered, flee
          self.move(random.randint(0, 3))
      else:  # Attack a random nearby agent
          target = random.choice(nearby_agents)
          if target.energy < self.energy:  # Only attack if stronger
              self.energy -= 10  # Fighting consumes energy
              target.alive = False  # Target dies


    def instinct(self, env, agents):
      # Get the agent's state
      state = self.get_state(env.food, agents)
      food_in_sight = [(fx, fy) for fx, fy in env.food if (fx - self.x) ** 2 + (fy - self.y) ** 2 <= 25]
      nearby_agents = [a for a in agents if (a.x - self.x) ** 2 + (a.y - self.y) ** 2 <= 9]

      # Priority-based instincts
      if self.energy < 30 and food_in_sight:  # Low energy, seek food
          self.seek_food(food_in_sight)

      elif self.energy > 50 and nearby_agents:  # High energy, consider reproduction
          self.seek_mate(nearby_agents, agents, self.generation)
      elif nearby_agents:  # If agents are nearby, decide whether to fight or flee
          self.aggressive_or_flee(nearby_agents)
      else:  # Otherwise, move randomly
          self.move(self.choose_action(state))

      # Update age, energy, and life status
      self.energy -= 1
      self.age += 1
      self.check_alive()

    def eat(self, food):
        if (self.x, self.y) in food:
            self.energy += 30
            food.remove((self.x, self.y))
            return True
        return False

    def reproduce(self, agents, generation):
        if self.energy > 50:
            mate = next((a for a in agents if a.x == self.x and a.y == self.y and a.gender != self.gender), None)
            if mate and mate.energy > 50:
                child_model = create_model(input_dim=6, output_dim=4)
                child_model.set_weights(self.model.get_weights())  # Inherit parent's weights
                if random.random() < 0.1:  # 10% mutation chance
                    mutate_model(child_model)

                child = Agent(self.x, self.y, random.choice(["M", "F"]), generation + 1)
                child.model = child_model
                agents.append(child)

                self.energy -= 20
                mate.energy -= 20
                return True
        return False

    def check_alive(self):
        if self.energy <= 0 or self.age > 120:
            self.alive = False

    def render(self, camera_x, camera_y):
        screen_x = (self.x * GRID_SIZE) - camera_x
        screen_y = (self.y * GRID_SIZE) - camera_y
        if 0 <= screen_x < SCREEN_WIDTH and 0 <= screen_y < SCREEN_HEIGHT:
            pygame.draw.circle(screen, BLUE if self else RED, (screen_x, screen_y), 5)
