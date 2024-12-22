from enum import Enum

from pydantic import BaseModel

class Ratio(Enum):
    SQAURE = "1:1"
    LANDSCAPE = "7:4"
    PORTRAIT = "4:7"


class ImageRequest(BaseModel):
    prompt: str
    n: int = 1
    ratio: Ratio = Ratio.SQAURE

class ImageResponse(BaseModel):
    urls: list[tuple[str, str]]