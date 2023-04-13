import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from collections import deque
from model.lstm_model import LSTMModel

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, num_layers=2):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = LSTMModel(state_size, hidden_size, num_layers, action_size)
        self.target_model = LSTMModel(state_size, hidden_size, num_layers, action_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(torch.tensor(state, dtype=torch.float).unsqueeze(0))
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.tensor(next_state, dtype=torch.float).unsqueeze(1).unsqueeze(0)
                target = (reward + self.gamma * torch.max(self.target_model(next_state_tensor)).item())
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(1).unsqueeze(0)
            target_f = self.model(state_tensor)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = F.mse_loss(self.model(torch.tensor(state, dtype=torch.float).unsqueeze(1).unsqueeze(0)), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
