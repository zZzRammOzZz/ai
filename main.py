from fastapi import FastAPI, File, UploadFile, Form
import numpy as np
import cv2
import torch
import json
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
from scipy.spatial.distance import cosine
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import ssl
from io import BytesIO
from PIL import Image

# Load environment variables
load_dotenv()
ssl._create_default_https_context = ssl._create_unverified_context

# Constants
THRESHOLD = 1.0
IMAGE_SIZE = (160, 160)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load models
face_recognition_model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all=True, device='cpu')


# FastAPI instance
app = FastAPI()

# Load known faces
def load_known_faces():
    """Load known faces from Supabase."""
    known_faces = {}
    try:
        response = supabase.table("employee").select("full_name, encoding").execute()
        for employee in response.data:
            encoding = np.array(json.loads(employee['encoding']))
            known_faces[employee['full_name']] = encoding
    except Exception as e:
        print(f"Error loading faces: {e}")
    return known_faces

known_faces = load_known_faces()

# Preprocessing function
def preprocess_image(image):
    """Preprocess an image for the FaceNet model."""
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0)

# Recognition function
def recognize_face(face):
    """Recognize the given face by comparing it with known faces."""
    img_tensor = preprocess_image(face)
    with torch.no_grad():
        embedding = face_recognition_model(img_tensor).numpy()

    min_distance = float('inf')
    recognized_user = None
    for name, known_embedding in known_faces.items():
        distance = cosine(embedding.flatten(), known_embedding.flatten())
        if distance < min_distance and distance < THRESHOLD:
            min_distance = distance
            recognized_user = name

    return recognized_user, min_distance

# API Endpoints

@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    """Recognize a face from an uploaded image."""
    image = Image.open(BytesIO(await file.read()))
    image = np.array(image)

    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        return {"error": "No face detected"}

    x1, y1, x2, y2 = map(int, boxes[0])
    face = image[y1:y2, x1:x2]

    if face is None or face.shape[0] == 0 or face.shape[1] == 0:
        return {"error": "Invalid face"}

    recognized_user, distance = recognize_face(Image.fromarray(face))

    return {"user": recognized_user, "distance": distance}

@app.post("/add/")
async def add_face(full_name: str = Form(...), store_id: str = Form(...), file: UploadFile = File(...)):
    """Add a new known face."""
    image = Image.open(BytesIO(await file.read()))
    image = np.array(image)

    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        return {"error": "No face detected"}

    x1, y1, x2, y2 = map(int, boxes[0])
    face = image[y1:y2, x1:x2]

    if face is None or face.shape[0] == 0 or face.shape[1] == 0:
        return {"error": "Invalid face"}

    img_tensor = preprocess_image(Image.fromarray(face))
    with torch.no_grad():
        embedding = face_recognition_model(img_tensor).numpy()
    encoding_json = json.dumps(embedding.tolist())

    try:
        response = supabase.table("employee").insert({
            "full_name": full_name,
            "store_id": store_id,
            "encoding": encoding_json
        }).execute()

        if response.data:
            known_faces[full_name] = embedding
            return {"message": f"{full_name} added successfully!"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/employees/")
async def get_all_faces():
    """Get all known faces."""
    return {"employees": list(known_faces.keys())}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
