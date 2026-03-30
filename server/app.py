from fastapi import FastAPI
from openenv.core.env_server.http_server import create_app
from server.environment import EpistemicNavEnvironment


from models import EpistemicAction, EpistemicObservation

app = create_app(lambda: EpistemicNavEnvironment(), EpistemicAction, EpistemicObservation)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == "__main__":
    main()
