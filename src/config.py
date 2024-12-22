import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE_URL = "https://api.openai.com/v1/images/generations"  # 실제 API 엔드포인트 확인 필요