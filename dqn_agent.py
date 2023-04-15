import random
import numpy as np
import torch
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size, memory_capacity, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, model, target_model, device):
        self.model = model
        self.target_model = target_model
        self.device = device
        self.memory = deque(maxlen=memory_capacity)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.model.to(device)
        self.target_model.to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def act(self, state, action_size):
        if np.random.rand() <= self.epsilon:
            return random.randrange(action_size)

        print("State:", state)

        state_tensor = torch.tensor(state[:-1], dtype=torch.float32).unsqueeze(0).to(self.device)
        sentiment_tensor = torch.tensor([state[-1]], dtype=torch.float32).unsqueeze(0).to(self.device)
        act_values = self.model(torch.cat((state_tensor, sentiment_tensor), dim=1))
        return torch.argmax(act_values).item()


    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)

            target = (reward + self.gamma * torch.max(self.target_model(next_state_tensor)).item()) if not done else reward
            target_f = self.model(state_tensor)
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state_tensor))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def retrain(self):
        self.target_model.load_state_dict(self.model.state_dict())
