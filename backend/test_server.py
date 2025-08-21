#!/usr/bin/env python3
"""
Simple test server for backend
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Adaptive Learning API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Adaptive Learning API is running!"}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "adaptive-learning-api"}

@app.get("/test")
def test():
    return {"test": "success", "message": "Backend is working correctly"}

if __name__ == "__main__":
    print("🚀 Starting test server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
