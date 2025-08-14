import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from keyword_recommendation_v1 import router as keyword_optimization_router
from data_fetcher import router as data_fetcher_router

# Load environment variables
load_dotenv()

# Get CORS origins from environment variable
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "https://nyx-ai-api.dev.nyx.today").split(",")

# Get host and port from environment variables
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5004"))

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(keyword_optimization_router)
app.include_router(data_fetcher_router)

@app.get("/")
async def root():
    return {"message": "Server is working fine"}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)

