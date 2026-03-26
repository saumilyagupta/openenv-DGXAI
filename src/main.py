# src/main.py
from fastapi import FastAPI
from openenv import OpenEnv
from .models import Action, Observation, State, Reward

class SaaSFactoryEnv(OpenEnv):
    def __init__(self):
        super().__init__()
        self.state: State = self._get_initial_state()

    def _get_initial_state(self) -> State:
        # This should eventually be more dynamic
        return State(
            current_task="generate-schema",
            workspace={},
            history=[]
        )

    def step(self, action: Action) -> (Observation, float, bool, dict):
        # Placeholder logic
        obs = Observation(files=[], output=f"Executed tool: {action.tool}", error=None)
        reward = Reward(score=0.0, explanation="Placeholder reward")
        done = False
        info = {}

        self.state.history.append({"action": action.dict(), "observation": obs.dict()})

        return obs, reward.score, done, info

    def reset(self) -> Observation:
        self.state = self._get_initial_state()
        # Return an initial observation
        return Observation(files=[], output="Environment has been reset.", error=None)

    def get_state(self) -> State:
        # In OpenEnv, the state() method is often called get_state() internally
        # to avoid conflicts with the 'state' property.
        return self.state

# Initialize the FastAPI app
app = FastAPI()

# Initialize our environment
env = SaaSFactoryEnv()

# Expose the OpenEnv endpoints
@app.post("/step", response_model=dict)
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {"observation": obs.dict(), "reward": reward, "done": done, "info": info}

@app.post("/reset", response_model=dict)
def reset():
    obs = env.reset()
    return {"observation": obs.dict()}

@app.get("/state", response_model=State)
def get_state():
    return env.get_state()

@app.get("/")
def read_root():
    return {"message": "Autonomous SaaS Factory OpenEnv is running"}
