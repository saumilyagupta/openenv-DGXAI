from fastapi import FastAPI, Response
import logging
import os
from openenv.core.env_server.http_server import create_app
from server.environment import EpistemicNavEnvironment


from models import EpistemicAction, EpistemicObservation

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

app = create_app(lambda: EpistemicNavEnvironment(), EpistemicAction, EpistemicObservation)


@app.get("/")
def root() -> dict:
    return {
        "name": "epistemic-nav",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/favicon.ico")
def favicon() -> Response:
    # Avoid noisy 404 logs from browser favicon requests.
    return Response(status_code=204)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == "__main__":
    main()
