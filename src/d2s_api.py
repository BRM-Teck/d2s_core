import io
import logging
from typing import List

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from d2s_processor.processor import ImageProcessor

app_logger = logging.getLogger(__name__)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_image_into_numpy_array(data):
    return np.array(Image.open(io.BytesIO(data)))


@app.post("/process-image/")
async def process_image(files: List[UploadFile]):
    arrays = [load_image_into_numpy_array(await file.read()) for file in files]
    processor = ImageProcessor(
        images=arrays, images_name=[file.filename for file in files]
    )
    data = processor.to_dict()
    app_logger.debug(data)
    return JSONResponse(content={"results": data})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
