from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from chat_agent import HybridChatbot  

# -----------------------
# Paths
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

# -----------------------
# Initialize
# -----------------------
app = FastAPI(title="Grocery Chatbot API")
bot = HybridChatbot()  # Load products & model

# Allow cross-origin requests (for frontend development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -----------------------
# Root route
# -----------------------
@app.get("/")
async def root():
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    return {"error": "index.html not found in /static"}

# -----------------------
# Health check
# -----------------------
@app.get("/ping")
async def ping():
    return {"status": "ok"}

# -----------------------
# Request model
# -----------------------
class UserQuery(BaseModel):
    text: str

# -----------------------
# Recommendation route
# -----------------------
@app.post("/recommend")
async def recommend(query: UserQuery):
    try:
        response = bot.get_response(query.text)

        # Normalize response into JSON structure
        if isinstance(response, str):
            return {"text": response, "products": []}
        elif isinstance(response, dict):
            return response
        else:
            return {"text": str(response), "products": []}

    except Exception as e:
        return {"text": "Sorry, something went wrong.", "error": str(e)}

# -----------------------
# Image recommendation route
# -----------------------
@app.post("/image-search")
async def image_search(file: UploadFile = File(...)):
    try:
        # Read the uploaded image into memory
        image_bytes = await file.read()

        # Pass the image to your chatbot's image search function
        response = bot.get_response_from_image(image_bytes)

        # Normalize response and add condition key for front-end
        if isinstance(response, str):
            return JSONResponse(content={
                "text": response,
                "products": [],
                "condition": "string_response"
            })
        elif isinstance(response, dict):
            # Ensure keys exist
            text = response.get("text", "")
            products = response.get("products", [])
            return JSONResponse(content={
                "text": text,
                "products": products,
                "condition": "dict_response"
            })
        else:
            return JSONResponse(content={
                "text": str(response),
                "products": [],
                "condition": "unknown_type"
            })

    except Exception as e:
        # Explicit error condition
        return JSONResponse(content={
            "text": "Sorry, something went wrong.",
            "products": [],
            "condition": "exception",
            "error": str(e)
        }, status_code=500)