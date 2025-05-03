import os
import uuid
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from text import generate_output
from image import generate_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root Endpoint
@app.get("/")
def home():
    return {"message": "FastAPI server is running!"}

# Define Input Schema
class StyleRequest(BaseModel):
    style_idea: str
    gender: str
    ethnicity: str
    age: str
    skin_color: str
    season: str
    accessories: str
    occasion: str

@app.post("/generate-outfit/")
async def generate_outfit(request: StyleRequest):
    try:
        logger.info("Received request: %s", request.dict())  # Logging request

        # Generate outfit description using model
        model_name = "tunedModels/outfitsuggestiongenerator-usqw4b296kfe"
        prompt = (
            f"Create an outfit including top, bottoms, and footwear based on: {request.style_idea} "
            f"for a {request.gender} from {request.ethnicity}, {request.age} years old, "
            f"having a {request.skin_color} complexion, to be worn in {request.season}, "
            f"with {request.accessories} accessories, for {request.occasion}. "
            f"Include matching footwear and ensure the model is looking at the camera."
        )
        
        outfit_description = generate_output(model_name, prompt)
        
        if not outfit_description:
            raise Exception("Text generation failed. Received empty response.")

        logger.info("Generated outfit description: %s", outfit_description)  # Logging generated description

        # Generate image
        image_filename = f"{uuid.uuid4()}.png"
        image_path = f"/tmp/{image_filename}"  # Use a temp directory for image storage in Render

        image_prompt = (
            f"A {request.gender} model, {request.age} years old, from {request.ethnicity} ethnicity, "
            f"with {request.skin_color} complexion, looking into the camera under perfect lighting, "
            f"wearing {outfit_description}. The image should be full body with a background that complements the outfit and model."
        )

        generated_image = generate_image(image_prompt)

        if generated_image is None:
            raise Exception("Image generation failed.")

        # Save the generated image to a temp directory
        generated_image.save(image_path)

        # For Render, the public URL will look like `https://your-app-name.onrender.com/generated_images/{image_filename}`
        image_url = f"https://fashionai-api.onrender.com/generated_images/{image_filename}"

        # Return the generated data with the image URL
        return {
            "outfit_description": outfit_description,
            "image_url": image_url
        }

    except Exception as e:
        logger.error("Error in /generate-outfit/: %s", str(e))  # Logging error
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/generated_images/{filename}")
async def get_generated_image(filename: str):
    # Serve image from /tmp folder (Render storage for temporary files)
    image_path = f"/tmp/{filename}"
    if os.path.exists(image_path):
        return FileResponse(image_path)
    raise HTTPException(status_code=404, detail="Image not found")
