"""
FastAPI server for RLSnake game inference.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import sys
import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Add RLSnake directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game import SnakeGame
from agent import DQNAgent

# Global game and agent storage
games = {}  # Store multiple game instances by session_id
agents = {}  # Store agents by model level
executor = ThreadPoolExecutor(max_workers=4)

# Model level to file mapping
MODEL_MAP = {
    'untrained': None,  # No model, random actions
    'beginner': 'model_episode_50.npz',  # Closest to ~100 episodes
    'intermediate': 'model_episode_250.npz',  # Closest to ~500 episodes
    'advanced': 'model_episode_1000.npz'  # Closest to ~2000 episodes
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global agents
    
    models_path = os.path.join(os.path.dirname(__file__), "models")
    
    if not os.path.exists(models_path):
        print(f"Warning: Models directory not found: {models_path}", file=sys.stderr, flush=True)
    else:
        print(f"Loading models from {models_path}...", file=sys.stderr, flush=True)
        start_time = time.time()
        
        # Load agents for each model level
        for level, model_file in MODEL_MAP.items():
            if model_file is None:
                # Untrained - create agent but don't load weights
                agents[level] = DQNAgent(state_size=20, action_size=3)
                agents[level].epsilon = 1.0  # Always random for untrained
                print(f"  {level}: Untrained (random actions)", file=sys.stderr, flush=True)
            else:
                agent = DQNAgent(state_size=20, action_size=3)
                model_path = os.path.join(models_path, model_file)
                if os.path.exists(model_path):
                    try:
                        agent.load(model_path)
                        agent.epsilon = 0.0  # No exploration during inference
                        agents[level] = agent
                        print(f"  {level}: Loaded {model_file}", file=sys.stderr, flush=True)
                    except Exception as e:
                        print(f"  Error loading {model_file}: {e}", file=sys.stderr, flush=True)
                        # Fallback to untrained
                        agents[level] = DQNAgent(state_size=20, action_size=3)
                        agents[level].epsilon = 1.0
                else:
                    print(f"  Warning: {model_file} not found, using untrained", file=sys.stderr, flush=True)
                    agents[level] = DQNAgent(state_size=20, action_size=3)
                    agents[level].epsilon = 1.0
        
        load_time = time.time() - start_time
        print(f"Models loaded successfully in {load_time:.2f}s!", file=sys.stderr, flush=True)
    
    yield
    
    # Shutdown
    print("Shutting down...", file=sys.stderr, flush=True)
    executor.shutdown(wait=False)

app = FastAPI(title="RLSnake API", version="1.0.0", lifespan=lifespan)

# CORS configuration
allowed_origins = os.getenv("ALLOWED_ORIGIN", "*")
if allowed_origins != "*":
    allowed_origins = [origin.strip() for origin in allowed_origins.split(",")]
    allowed_origins.append("http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if isinstance(allowed_origins, list) else ["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

class StartRequest(BaseModel):
    model_level: str  # 'untrained', 'beginner', 'intermediate', 'advanced'
    session_id: str = None  # Optional session ID for multiple games

class StepRequest(BaseModel):
    session_id: str

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "models_loaded": len(agents) > 0,
        "service": "RLSnake API"
    }

@app.get("/health")
async def health():
    """Health check with model status."""
    return {
        "status": "healthy",
        "models_loaded": len(agents) > 0,
        "available_levels": list(agents.keys()),
        "service": "RLSnake API"
    }

@app.post("/start")
async def start_game(request: StartRequest):
    """Start a new game with the specified model."""
    if request.model_level not in MODEL_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_level. Must be one of: {list(MODEL_MAP.keys())}"
        )
    
    if request.model_level not in agents:
        raise HTTPException(
            status_code=503,
            detail=f"Model {request.model_level} not loaded. Please check server logs."
        )
    
    session_id = request.session_id or f"game_{int(time.time() * 1000)}"
    
    # Run game initialization in thread pool
    loop = asyncio.get_event_loop()
    game_state = await loop.run_in_executor(
        executor,
        _initialize_game,
        session_id,
        request.model_level
    )
    
    return game_state

def _initialize_game(session_id: str, model_level: str):
    """Initialize a new game (runs in thread pool)."""
    game = SnakeGame(width=20, height=20, cell_size=20)
    state = game.reset()  # Returns numpy array
    games[session_id] = {
        'game': game,
        'model_level': model_level,
        'state': state  # Store as numpy array internally
    }
    
    return {
        "session_id": session_id,
        "snake": game.snake,
        "food": game.food,
        "score": game.score,
        "moves": game.steps,
        "game_over": game.done,
        "direction": game.direction.name
    }

@app.post("/step")
async def step_game(request: StepRequest):
    """Get next game step (AI makes move, returns updated state)."""
    if request.session_id not in games:
        raise HTTPException(
            status_code=404,
            detail=f"Game session {request.session_id} not found. Start a new game first."
        )
    
    game_data = games[request.session_id]
    game = game_data['game']
    model_level = game_data['model_level']
    agent = agents[model_level]
    
    # Convert stored state back to numpy array if it's a list
    stored_state = game_data['state']
    if isinstance(stored_state, list):
        stored_state = np.array(stored_state, dtype=np.float32)
    
    # Run game step in thread pool
    loop = asyncio.get_event_loop()
    game_state = await loop.run_in_executor(
        executor,
        _step_game,
        game,
        agent,
        stored_state
    )
    
    # Update stored state (keep as numpy array internally, convert to list only for JSON)
    game_data['state'] = np.array(game_state['state'], dtype=np.float32) if isinstance(game_state['state'], list) else game_state['state']
    
    # Remove game if it's over
    if game_state['game_over']:
        del games[request.session_id]
    
    return game_state

def _step_game(game: SnakeGame, agent: DQNAgent, current_state):
    """Execute one game step (runs in thread pool)."""
    # Convert list back to numpy array if needed (state is stored as list for JSON)
    if isinstance(current_state, list):
        current_state = np.array(current_state, dtype=np.float32)
    
    if game.done:
        return {
            "snake": game.snake,
            "food": game.food,
            "score": game.score,
            "moves": game.steps,
            "game_over": True,
            "direction": game.direction.name,
            "state": current_state.tolist() if isinstance(current_state, np.ndarray) else current_state
        }
    
    # Agent chooses action
    action = agent.act(current_state)
    
    # Execute action
    next_state, reward, done = game.step(action)
    
    return {
        "snake": game.snake,
        "food": game.food,
        "score": game.score,
        "moves": game.steps,
        "game_over": done,
        "direction": game.direction.name,
        "action": int(action),
        "reward": float(reward),
        "state": next_state.tolist()  # Convert numpy array to list for JSON
    }

@app.post("/reset")
async def reset_game(request: StepRequest):
    """Reset an existing game."""
    if request.session_id not in games:
        raise HTTPException(
            status_code=404,
            detail=f"Game session {request.session_id} not found."
        )
    
    game_data = games[request.session_id]
    game = game_data['game']
    
    # Run reset in thread pool
    loop = asyncio.get_event_loop()
    game_state = await loop.run_in_executor(
        executor,
        _reset_game,
        game
    )
    
    # Store state as numpy array (convert from list)
    game_data['state'] = np.array(game_state['state'], dtype=np.float32)
    
    return game_state

def _reset_game(game: SnakeGame):
    """Reset game to initial state (runs in thread pool)."""
    state = game.reset()
    return {
        "snake": game.snake,
        "food": game.food,
        "score": game.score,
        "moves": game.steps,
        "game_over": False,
        "direction": game.direction.name,
        "state": state.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
