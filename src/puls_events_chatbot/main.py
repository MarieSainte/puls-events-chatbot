from contextlib import asynccontextmanager
from fastapi import FastAPI
import threading
import uvicorn
import gradio as gr
from puls_events_chatbot.controllers import chatbot_controller
from puls_events_chatbot.gradio_interface import launch_frontend
from puls_events_chatbot.services.chatbot import init

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initialisation BDD vectorielle...")
    init()         

    frontend = launch_frontend()
    thread = threading.Thread(
        target=frontend.launch,
        kwargs={"server_name": "0.0.0.0", "server_port": 7860, "quiet": True},
        daemon=True,
    )
    thread.start()
    yield            
    print("Arrêt de l'application")

app = FastAPI(
    title = "Chatbot",
    description="API pour communiquer avec le chatbot de puls events",
    version="1.0",
    docs_url="/swagger",
    redoc_url="/redoc",
    lifespan=lifespan,
)


app.include_router(chatbot_controller.router)

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "message": "The application is running."}

    