import numpy as np
import pygame
from enum import Enum
from typing import Tuple, List, Optional

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class SnakeGame:
    def __init__(self, width: int = 20, height: int = 20, cell_size: int = 20):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset game to initial state, return state vector"""
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = Direction.RIGHT
        self.food = self._spawn_food()
        self.score = 0
        self.steps = 0
        self.done = False
        return self.get_state()
    
    def _spawn_food(self) -> Tuple[int, int]:
        """Spawn food at random location not occupied by snake"""
        while True:
            food = (np.random.randint(0, self.width), np.random.randint(0, self.height))
            if food not in self.snake:
                return food
    
    def get_state(self) -> np.ndarray:
        """Get current state as feature vector for RL agent"""
        head_x, head_y = self.snake[0]
        
        # Danger detection (immediate collision risk)
        danger_straight = self._check_danger(self.direction)
        danger_right = self._check_danger(self._turn_right())
        danger_left = self._check_danger(self._turn_left())
        
        # Vision rays - what the agent can see in each direction
        # Forward vision (direction of travel)
        f_wall, f_body, f_food, f_food_vis = self._cast_vision_ray(self.direction)
        # Left vision
        l_wall, l_body, l_food, l_food_vis = self._cast_vision_ray(self._turn_left())
        # Right vision
        r_wall, r_body, r_food, r_food_vis = self._cast_vision_ray(self._turn_right())
        
        # Direction one-hot encoding
        dir_up = 1 if self.direction == Direction.UP else 0
        dir_down = 1 if self.direction == Direction.DOWN else 0
        dir_left = 1 if self.direction == Direction.LEFT else 0
        dir_right = 1 if self.direction == Direction.RIGHT else 0
        
        # Snake length (normalized)
        snake_length = len(self.snake) / (self.width * self.height)
        
        return np.array([
            # Immediate dangers
            danger_straight, danger_right, danger_left,
            # Forward vision (direction of travel)
            f_wall, f_body, f_food, f_food_vis,
            # Left vision
            l_wall, l_body, l_food, l_food_vis,
            # Right vision
            r_wall, r_body, r_food, r_food_vis,
            # Direction and length
            dir_up, dir_down, dir_left, dir_right, snake_length
        ], dtype=np.float32)
    
    def _check_danger(self, direction: Direction) -> float:
        """Check if moving in direction would cause collision"""
        dx, dy = direction.value
        head_x, head_y = self.snake[0]
        next_pos = (head_x + dx, head_y + dy)
        
        # Check wall collision
        if next_pos[0] < 0 or next_pos[0] >= self.width or \
           next_pos[1] < 0 or next_pos[1] >= self.height:
            return 1.0
        
        # Check body collision
        if next_pos in self.snake[:-1]:
            return 1.0
        
        return 0.0
    
    def _cast_vision_ray(self, direction: Direction) -> Tuple[float, float, float, float]:
        """
        Cast a ray in given direction and return what it sees.
        Returns: (wall_distance, body_distance, food_distance, food_visible)
        All distances normalized (0-1), 1.0 means not found/visible
        """
        dx, dy = direction.value
        head_x, head_y = self.snake[0]
        
        wall_dist = 1.0
        body_dist = 1.0
        food_dist = 1.0
        food_visible = 0.0
        
        # Check if food is visible in this direction (on the ray)
        if dx != 0:  # Horizontal movement
            if self.food[1] == head_y:  # Same row
                if (dx > 0 and self.food[0] > head_x) or (dx < 0 and self.food[0] < head_x):
                    food_dist = abs(self.food[0] - head_x) / max(self.width, self.height)
                    food_visible = 1.0
        else:  # Vertical movement
            if self.food[0] == head_x:  # Same column
                if (dy > 0 and self.food[1] > head_y) or (dy < 0 and self.food[1] < head_y):
                    food_dist = abs(self.food[1] - head_y) / max(self.width, self.height)
                    food_visible = 1.0
        
        # Cast ray from head in direction to find walls and body
        max_dist = max(self.width, self.height)
        for step in range(1, max_dist + 1):
            x = head_x + dx * step
            y = head_y + dy * step
            
            # Check if out of bounds (wall)
            if wall_dist == 1.0 and (x < 0 or x >= self.width or y < 0 or y >= self.height):
                wall_dist = step / max_dist
                break  # Can't see past wall
            
            # Check if hit body
            if body_dist == 1.0 and (x, y) in self.snake[:-1]:
                body_dist = step / max_dist
                # Don't break - might see food behind body
            
            # If we found everything or hit wall, we're done
            if wall_dist < 1.0:
                break
        
        return wall_dist, body_dist, food_dist, food_visible
    
    def _turn_left(self) -> Direction:
        """Get direction after turning left"""
        turns = {
            Direction.UP: Direction.LEFT,
            Direction.LEFT: Direction.DOWN,
            Direction.DOWN: Direction.RIGHT,
            Direction.RIGHT: Direction.UP
        }
        return turns[self.direction]
    
    def _turn_right(self) -> Direction:
        """Get direction after turning right"""
        turns = {
            Direction.UP: Direction.RIGHT,
            Direction.RIGHT: Direction.DOWN,
            Direction.DOWN: Direction.LEFT,
            Direction.LEFT: Direction.UP
        }
        return turns[self.direction]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute action: 0=straight, 1=right, 2=left
        Returns: (next_state, reward, done)
        """
        if self.done:
            return self.get_state(), 0, True
        
        self.steps += 1
        
        # Update direction based on action
        if action == 1:  # Right turn
            self.direction = self._turn_right()
        elif action == 2:  # Left turn
            self.direction = self._turn_left()
        # action == 0 means continue straight (no change)
        
        # Move snake
        dx, dy = self.direction.value
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)
        
        # Check collisions
        if new_head[0] < 0 or new_head[0] >= self.width or \
           new_head[1] < 0 or new_head[1] >= self.height or \
           new_head in self.snake:
            self.done = True
            return self.get_state(), -10, True
        
        self.snake.insert(0, new_head)
        
        # Check if food eaten
        reward = 0
        if new_head == self.food:
            self.score += 1
            reward = 100  # Very high reward for eating food
            self.food = self._spawn_food()
        else:
            self.snake.pop()
            # Reward only for significant progress (reduces exploitation)
            old_dist = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
            new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            dist_change = old_dist - new_dist
            if dist_change > 0.5:  # Only reward if moved significantly closer
                reward = 0.5
            elif dist_change < -0.5:  # Penalty for moving significantly away
                reward = -0.3
        
        # Negative reward per step (increases with snake length to prevent spinning)
        reward -= 0.05 * (1 + len(self.snake) / 20)  # More penalty when longer
        
        # Penalty for too many steps without eating
        if self.steps > 100 * (self.score + 1):
            self.done = True
            reward = -10
        
        return self.get_state(), reward, self.done
    
    def render(self, screen: Optional[pygame.Surface] = None) -> Optional[np.ndarray]:
        """Render game. If screen provided, draw to it. Otherwise return array"""
        if screen is None:
            # Return state array for headless mode
            return None
        
        screen.fill((20, 20, 20))
        
        # Draw grid
        for x in range(self.width):
            for y in range(self.height):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                 self.cell_size, self.cell_size)
                pygame.draw.rect(screen, (30, 30, 30), rect, 1)
        
        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                             self.cell_size, self.cell_size)
            pygame.draw.rect(screen, color, rect)
        
        # Draw food
        food_rect = pygame.Rect(self.food[0] * self.cell_size,
                              self.food[1] * self.cell_size,
                              self.cell_size, self.cell_size)
        pygame.draw.rect(screen, (255, 0, 0), food_rect)
        
        return None

