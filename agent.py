import numpy as np
import random
from collections import deque
from typing import Tuple

class DQNAgent:
    def __init__(self, state_size: int = 20, action_size: int = 3, 
                 learning_rate: float = 0.0005, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, memory_size: int = 10000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Main network (learns)
        self.weights1 = np.random.randn(state_size, 128) * 0.1
        self.bias1 = np.zeros((1, 128))
        self.weights2 = np.random.randn(128, action_size) * 0.1
        self.bias2 = np.zeros((1, action_size))
        
        # Target network (stable, updated periodically)
        self.target_weights1 = self.weights1.copy()
        self.target_bias1 = self.bias1.copy()
        self.target_weights2 = self.weights2.copy()
        self.target_bias2 = self.bias2.copy()
        
        self.update_target_freq = 100  # Update target network every N steps
        self.train_step = 0
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU derivative"""
        return (x > 0).astype(float)
    
    def predict(self, state: np.ndarray, use_target: bool = False) -> np.ndarray:
        """Predict Q-values for given state"""
        state = state.reshape(1, -1)
        if use_target:
            z1 = np.dot(state, self.target_weights1) + self.target_bias1
            a1 = self._relu(z1)
            z2 = np.dot(a1, self.target_weights2) + self.target_bias2
        else:
            z1 = np.dot(state, self.weights1) + self.bias1
            a1 = self._relu(z1)
            z2 = np.dot(a1, self.weights2) + self.bias2
        return z2[0]
    
    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.predict(state)
        return np.argmax(q_values)
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size: int = 32):
        """Train on batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q values (from main network)
        current_q = self.predict_batch(states, use_target=False)
        
        # Next Q values (from target network for stability)
        next_q = self.predict_batch(next_states, use_target=True)
        target_q = current_q.copy()
        
        for i in range(batch_size):
            if dones[i]:
                target_q[i][actions[i]] = rewards[i]
            else:
                target_q[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        # Train network
        self._train_batch(states, target_q)
        
        self.train_step += 1
        
        # Update target network periodically (reduces instability)
        if self.train_step % self.update_target_freq == 0:
            self.target_weights1 = self.weights1.copy()
            self.target_bias1 = self.bias1.copy()
            self.target_weights2 = self.weights2.copy()
            self.target_bias2 = self.bias2.copy()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def predict_batch(self, states: np.ndarray, use_target: bool = False) -> np.ndarray:
        """Predict Q-values for batch of states"""
        if use_target:
            z1 = np.dot(states, self.target_weights1) + self.target_bias1
            a1 = self._relu(z1)
            z2 = np.dot(a1, self.target_weights2) + self.target_bias2
        else:
            z1 = np.dot(states, self.weights1) + self.bias1
            a1 = self._relu(z1)
            z2 = np.dot(a1, self.weights2) + self.bias2
        return z2
    
    def _train_batch(self, states: np.ndarray, target_q: np.ndarray):
        """Train network on batch using gradient descent"""
        batch_size = states.shape[0]
        
        # Forward pass
        z1 = np.dot(states, self.weights1) + self.bias1
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.weights2) + self.bias2
        
        # Backward pass
        dz2 = (z2 - target_q) / batch_size
        dw2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        da1 = np.dot(dz2, self.weights2.T)
        dz1 = da1 * self._relu_derivative(z1)
        dw1 = np.dot(states.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.weights2 -= self.learning_rate * dw2
        self.bias2 -= self.learning_rate * db2
        self.weights1 -= self.learning_rate * dw1
        self.bias1 -= self.learning_rate * db1
    
    def save(self, filepath: str):
        """Save agent weights"""
        np.savez(filepath, 
                weights1=self.weights1, bias1=self.bias1,
                weights2=self.weights2, bias2=self.bias2,
                target_weights1=self.target_weights1, target_bias1=self.target_bias1,
                target_weights2=self.target_weights2, target_bias2=self.target_bias2,
                epsilon=self.epsilon, train_step=self.train_step)
    
    def load(self, filepath: str):
        """Load agent weights"""
        data = np.load(filepath, allow_pickle=True)
        self.weights1 = data['weights1']
        self.bias1 = data['bias1']
        self.weights2 = data['weights2']
        self.bias2 = data['bias2']
        # Try to load target network, fallback to copying main network
        try:
            self.target_weights1 = data['target_weights1']
            self.target_bias1 = data['target_bias1']
            self.target_weights2 = data['target_weights2']
            self.target_bias2 = data['target_bias2']
        except:
            self.target_weights1 = self.weights1.copy()
            self.target_bias1 = self.bias1.copy()
            self.target_weights2 = self.weights2.copy()
            self.target_bias2 = self.bias2.copy()
        self.epsilon = float(data['epsilon'])
        try:
            self.train_step = int(data['train_step'])
        except:
            self.train_step = 0

