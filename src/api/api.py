from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['GET', 'POST', 'PUT', 'DELETE'],
    allow_headers=['*'],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@app.post('/upload/')
async def upload_file(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))

        top1_image = image
        top2_image = image
        top3_image = image

        top1_image_base64 = image_to_base64(top1_image)
        top2_image_base64 = image_to_base64(top2_image)
        top3_image_base64 = image_to_base64(top3_image)

        response_content = {
            'top1_image': top1_image_base64,
            'top2_image': top2_image_base64,
            'top3_image': top3_image_base64
        }
        
        logger.info("Successful processing and response generation.")
        return JSONResponse(content=response_content)

    except Exception as e:
        logger.error(f"Error processing the file: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)