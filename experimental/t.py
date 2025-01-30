import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Constants
grid_size = 10
tile_size = 40
screen_size = grid_size * tile_size

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Neural Network for AI
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class BattleshipGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_size * 2, screen_size))
        pygame.display.set_caption("Battleship Game")
        self.font = pygame.font.Font(None, 36)

        # Boards
        self.player_board = np.zeros((grid_size, grid_size), dtype=int)
        self.ai_board = np.zeros((grid_size, grid_size), dtype=int)
        self.player_visible = np.full((grid_size, grid_size), -1)
        self.ai_visible = np.full((grid_size, grid_size), -1)

        # Game state
        self.player_turn = True
        self.running = True
        self.place_phase = True
        self.ships_to_place = [5, 4, 3, 3, 2]
        self.current_ship = 0

        # AI memory and RL parameters
        self.ai_guesses = set()
        self.state_size = grid_size * grid_size
        self.action_size = grid_size * grid_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.learning_rate = 0.001

        # DQN Model
        self.model = DQN(self.state_size, self.action_size)
        self.target_model = DQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_model()

        self.memory = []
        self.batch_size = 32

        self.train_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def draw_grid(self, x_offset, y_offset, board, visible):
        for row in range(grid_size):
            for col in range(grid_size):
                x = x_offset + col * tile_size
                y = y_offset + row * tile_size

                if visible[row, col] == 1:
                    color = GREEN if board[row, col] == 1 else RED
                else:
                    color = WHITE

                pygame.draw.rect(self.screen, color, (x, y, tile_size, tile_size))
                pygame.draw.rect(self.screen, BLUE, (x, y, tile_size, tile_size), 1)

    def place_ship(self, board, pos, length, horizontal):
        row, col = pos
        if horizontal:
            if col + length > grid_size:
                return False
            if np.any(board[row, col:col+length] == 1):
                return False
            board[row, col:col+length] = 1
        else:
            if row + length > grid_size:
                return False
            if np.any(board[row:row+length, col] == 1):
                return False
            board[row:row+length, col] = 1
        return True

    def train_model(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            self.memory.pop(0)

    def get_state(self, board):
        return board.flatten()

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def ai_turn(self):
        state = self.get_state(self.player_visible)
        action = self.choose_action(state)

        row, col = divmod(action, grid_size)
        if (row, col) not in self.ai_guesses:
            self.ai_guesses.add((row, col))

            reward = 0
            if self.player_board[row, col] == 1:
                self.player_visible[row, col] = 1
                reward = 1
                print("AI hit!")
            else:
                self.player_visible[row, col] = 0
                print("AI miss!")

            next_state = self.get_state(self.player_visible)
            done = np.all(self.player_visible[self.player_board == 1] == 1)

            self.remember(state, action, reward, next_state, done)

            if done:
                self.update_target_model()

            self.train_model()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def handle_click(self, pos):
        x, y = pos
        if self.place_phase:
            board_x = x // tile_size
            board_y = y // tile_size

            if self.current_ship < len(self.ships_to_place):
                ship_length = self.ships_to_place[self.current_ship]
                if self.place_ship(self.player_board, (board_y, board_x), ship_length, horizontal=True):
                    self.current_ship += 1
                    if self.current_ship == len(self.ships_to_place):
                        self.place_phase = False
                        self.ai_place_ships()
        else:
            board_x = (x - screen_size) // tile_size
            board_y = y // tile_size
            if 0 <= board_x < grid_size:
                if self.ai_visible[board_y, board_x] == -1:
                    self.ai_visible[board_y, board_x] = 1 if self.ai_board[board_y, board_x] == 1 else 0
                    self.player_turn = False

    def ai_place_ships(self):
        for ship_length in self.ships_to_place:
            placed = False
            while not placed:
                row = random.randint(0, grid_size - 1)
                col = random.randint(0, grid_size - 1)
                horizontal = random.choice([True, False])
                placed = self.place_ship(self.ai_board, (row, col), ship_length, horizontal)

    def run(self):
        while self.running:
            self.screen.fill(WHITE)

            self.draw_grid(0, 0, self.player_board, self.player_visible)
            self.draw_grid(screen_size, 0, self.ai_board, self.ai_visible)

            for event in pygame.event.get():
                print(event)
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    print("a", self.player_turn)
                    if self.player_turn:
                        self.handle_click(event.pos)

            if not self.player_turn and not self.place_phase:
                self.ai_turn()
                self.player_turn = True

            pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    game = BattleshipGame()
    game.run()
