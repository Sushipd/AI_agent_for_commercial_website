from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
            return {"text": response, "products": None}
        elif isinstance(response, dict):
            return response
        else:
            return {"text": str(response), "products": None}

    except Exception as e:
        return {"text": "Sorry, something went wrong.", "error": str(e)}
