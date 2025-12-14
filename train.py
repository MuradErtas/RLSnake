import numpy as np
import pygame
from game import SnakeGame
from agent import DQNAgent
import matplotlib.pyplot as plt
from collections import deque

class Trainer:
    def __init__(self, episodes: int = 1000, render: bool = False, 
                 render_every: int = 100, save_every: int = 1000):
        self.episodes = episodes
        self.render = render
        self.render_every = render_every
        self.save_every = save_every
        
        self.game = SnakeGame(width=20, height=20)
        self.agent = DQNAgent(state_size=20, action_size=3)
        
        # Tracking metrics
        self.scores = []
        self.avg_scores = deque(maxlen=100)
        
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.game.width * self.game.cell_size,
                 self.game.height * self.game.cell_size)
            )
            pygame.display.set_caption("RL Snake Training")
            self.clock = pygame.time.Clock()
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Initial epsilon: {self.agent.epsilon:.3f}")
        
        for episode in range(self.episodes):
            state = self.game.reset()
            total_reward = 0
            steps = 0
            
            should_render = self.render and (episode % self.render_every == 0 or episode == self.episodes - 1)
            
            while not self.game.done:
                # Handle pygame events
                if should_render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            return
                
                # Agent chooses action
                action = self.agent.act(state)
                
                # Execute action
                next_state, reward, done = self.game.step(action)
                total_reward += reward
                steps += 1
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, done)
                
                state = next_state
                
                # Train agent multiple times per episode (more frequent learning)
                if len(self.agent.memory) > 32 and steps % 4 == 0:
                    self.agent.replay(batch_size=32)
                
                # Render if needed
                if should_render:
                    self.game.render(self.screen)
                    pygame.display.flip()
                    self.clock.tick(10)  # Slow down for visibility
            
            # Final training step for episode
            if len(self.agent.memory) > 32:
                self.agent.replay(batch_size=32)
            
            # Track metrics
            score = self.game.score
            self.scores.append(score)
            self.avg_scores.append(score)
            avg_score = np.mean(self.avg_scores) if self.avg_scores else 0
            
            # Print progress
            if episode % 10 == 0:
                print(f"Episode {episode:4d} | Score: {score:3d} | "
                      f"Avg Score: {avg_score:5.2f} | "
                      f"Epsilon: {self.agent.epsilon:.3f} | Steps: {steps}")
            
            # Save model periodically
            #if episode > 0 and episode % self.save_every == 0:
            #    self.agent.save(f"models/model_episode_{episode}.npz")
            #    print(f"Model saved at episode {episode}")
        
        # Final save
        self.agent.save("models/model_final.npz")
        print("Training complete! Final model saved.")
        
        if self.render:
            pygame.quit()
    
    def plot_scores(self):
        """Plot training progress"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.scores, alpha=0.3, label='Score')
        if len(self.scores) >= 100:
            moving_avg = [np.mean(self.scores[max(0, i-100):i+1]) 
                         for i in range(len(self.scores))]
            plt.plot(moving_avg, label='Moving Avg (100)')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.hist(self.scores, bins=20, edgecolor='black')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/training_progress.png')
        print("Training plot saved as 'training_progress.png'")
        plt.show()

if __name__ == "__main__":
    trainer = Trainer(episodes=1000, render=False, render_every=50)
    trainer.train()
    trainer.plot_scores()

