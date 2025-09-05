from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from typing import List
import uvicorn

# --- Import de TES fonctions existantes ---
from backend.index_data import index_documents  # Ta fonction d'indexation
from backend.generator_new import generate_from_query  # Ta fonction de génération

# --- Configuration FastAPI ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise toutes les origines (à restreindre en production)
    allow_methods=["*"],  # Autorise toutes les méthodes
    allow_headers=["*"],
)

# --- Dossier pour les uploads ---
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Endpoint pour uploader et indexer des documents.
    """
    file_paths = []
    try:
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(file_path)

        # --- Appel à TA fonction d'indexation existante ---
        collection = index_documents(file_paths)  # Utilise ta fonction index_documents()
        return {
            "status": "success",
            "message": f"{len(file_paths)} documents indexés avec succès.",
            "collection_size": collection.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'indexation : {str(e)}")

@app.post("/analyze")
async def analyze_query(request: Request):
    """
    Endpoint pour analyser une requête utilisateur.
    """
    try:
        data = await request.json()
        query = data["query"]

        # --- Appel à TA fonction de génération existante ---
        result = generate_from_query(query)  # Utilise ta fonction generate_from_query()
        return {"status": "success", "summary": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse : {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)