from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from ui_askme import demo as askme
from ui_vp import demo as vp
import gradio as gr
import os

# Define the folder path for outputs
folder_path = "outputs"

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created successfully.")
else:
    print(f"Folder '{folder_path}' already exists.")

# Initialize the FastAPI application
parent = FastAPI()

# Mount static files (CSS, JS, images) under the "/static" path
parent.mount("/static", StaticFiles(directory="static"), name="static")

# Define the root endpoint that serves the index.html file
@parent.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    # Mount Gradio app for "ask_me" under the "/ask_me" path
    fast_askme = gr.mount_gradio_app(parent, askme, path="/ask_me")

    # Mount Gradio app for "video_processing" under the "/video_processing" path
    fast_vp = gr.mount_gradio_app(parent, vp, path="/video_processing")

    # Mount the Gradio apps to the FastAPI application
    parent.mount("/ask_me", fast_askme)
    parent.mount("/video_processing", fast_vp)

    # Run the FastAPI application with Uvicorn
    uvicorn.run(parent, host="0.0.0.0", port=8000)
