from fastapi import FastAPI, HTTPException
from src.models.request_response import ImageRequest, ImageResponse
from .dalle_api import generate_image

app = FastAPI(title="DALL·E 3 Inference API")

@app.post("/generate", response_model=ImageResponse)
async def generate_endpoint(req: ImageRequest):
    try:
        urls = generate_image(req.prompt, req.n, req.ratio)
        return ImageResponse(urls=urls)
    except Exception as e:
        # 에러 처리를 좀 더 정교하게 할 수 있음
        raise HTTPException(status_code=500, detail=str(e))