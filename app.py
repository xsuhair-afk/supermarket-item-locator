from fastapi import FastAPI, UploadFile, File, Form
from feature_extractor import VGG16Extractor
from vector_db import VectorDB
import json
import uvicorn
from contextlib import asynccontextmanager

db = VectorDB()
extractor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the machine learning model on startup
    global extractor
    print("Loading VGG16 Model... This might take a moment.")
    extractor = VGG16Extractor()
    print("Model loaded successfully.")
    yield
    # Clean up on shutdown if needed
    print("Shutting down CBIR backend.")

app = FastAPI(title="Harvest Concierge CBIR Backend", lifespan=lifespan)

@app.post("/index")
async def index_image(id: str = Form(...), metadata: str = Form(...), file: UploadFile = File(...)):
    """
    Index a new image into the vector database.
    metadata should be a JSON string.
    """
    img_bytes = await file.read()
    features = extractor.extract(img_bytes)
    try:
        meta_dict = json.loads(metadata)
    except json.JSONDecodeError:
        meta_dict = {"raw": metadata}
        
    db.insert(id, meta_dict, features)
    return {"message": "Indexed successfully", "id": id}

@app.post("/search")
async def search_image(file: UploadFile = File(...), top_k: int = 5):
    """
    Search for similar images using CBIR.
    Returns the top_k closest matches utilizing multi-metric similarity checks.
    """
    img_bytes = await file.read()
    features = extractor.extract(img_bytes)
    results = db.search(features, top_k=top_k)
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
