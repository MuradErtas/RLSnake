import pygame
import sys
from game import SnakeGame
from agent import DQNAgent
import argparse

def play_with_ai(model_path: str = None, speed: int = 10):
    """Play game with trained AI agent"""
    pygame.init()
    
    game = SnakeGame(width=20, height=20, cell_size=30)
    agent = DQNAgent(state_size=20, action_size=3)
    
    if model_path:
        try:
            agent.load(f"models/{model_path}")
            print(f"Loaded model from {model_path}")
            print(f"Epsilon: {agent.epsilon:.3f}")
        except:
            print(f"Could not load {model_path}, using untrained agent")
    
    screen = pygame.display.set_mode(
        (game.width * game.cell_size, game.height * game.cell_size)
    )
    pygame.display.set_caption("RL Snake - AI Playing")
    clock = pygame.time.Clock()
    
    state = game.reset()
    score = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # AI chooses action
        action = agent.act(state)
        
        # Execute action
        next_state, reward, done = game.step(action)
        state = next_state
        
        if done:
            print(f"Game Over! Final Score: {game.score}")
            state = game.reset()
            score = 0
        
        # Render
        game.render(screen)
        
        # Display score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {game.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(speed)
    
    pygame.quit()

def train_mode(episodes: int = 1000, render: bool = False):
    """Run training mode"""
    from train import Trainer
    trainer = Trainer(episodes=episodes, render=render, render_every=50)
    trainer.train()
    trainer.plot_scores()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Snake Game")
    parser.add_argument("--mode", choices=["train", "play"], default="train",
                       help="Mode: train or play")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to model file for playing")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of training episodes")
    parser.add_argument("--render", action="store_true",
                       help="Render during training")
    parser.add_argument("--speed", type=int, default=10,
                       help="Game speed for playing mode")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_mode(episodes=args.episodes, render=args.render)
    else:
        play_with_ai(model_path=args.model, speed=args.speed)

