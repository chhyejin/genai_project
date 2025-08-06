from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pydantic import BaseModel, Field, root_validator
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional
from datetime import datetime
from PIL import Image
import torch
import os
import shutil
import json 
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 환경 설정 (환경변수에서 읽어오기)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 경로 설정 (환경변수에서 읽어오기)
BASE_DIR = os.getenv("BASE_DIR", os.getcwd())  # 기본값은 현재 작업 디렉토리
STATIC_DIR = os.path.join(BASE_DIR, os.getenv("STATIC_DIR", "static"))
UPLOAD_FOLDER = os.path.join(STATIC_DIR, os.getenv("UPLOAD_FOLDER", "uploads"))
TEMPLATE_DIR = os.path.join(BASE_DIR, os.getenv("TEMPLATE_DIR", "templates"))

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 모델 로딩 (환경변수에서 모델 ID 읽어오기)
model_id = os.getenv("MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# LangChain 모델
class TransactionInfo(BaseModel):
    name: str = Field(description="이름")
    ssn_front: str = Field(description="주민등록번호 앞자리(생년월일 6자리)")
    address: str = Field(description="주소")
    ssn_back_first: str = Field(description="주민등록번호 뒷자리에서 성별을 구분하는 첫 번째 숫자 즉, 8번째 자리 숫자 (예: 000000-1XXXX에서 1)")
    gender: Optional[str] = Field(default=None, description="성별")

    @root_validator(pre=True)
    def set_gender(cls, values):
        code = values.get("ssn_back_first")
        if code in ["1", "3"]:
            values["gender"] = "남성"
        elif code in ["2", "4"]:
            values["gender"] = "여성"
        else:
            values["gender"] = "알 수 없음"
        return values

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm_structured = llm.with_structured_output(TransactionInfo)


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    try:
        filename = datetime.now().strftime("%Y%m%d%H%M%S_") + file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": file_path},
                    {"type": "text", "text": "Extract texts in markdown format"},
                ],
            }
        ]

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        inputs = processor(
            text=[prompt],
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        ).to("cuda")  

        generated = model.generate(**inputs, max_new_tokens=256)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        text_output = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        # LangChain 구조화
        try:
            result = llm_structured.invoke(text_output)
            result_dict = result.dict()
        except Exception as e:
            result_dict = {"error": f"구조화 실패: {e}"}

        # JSON 문자열로 변환 (ensure_ascii=False로 한글 깨지지 않도록)
        result_json = json.dumps(result_dict, indent=2, ensure_ascii=False)

        return templates.TemplateResponse("result.html", {
            "request": request,
            "image_url": "/static/uploads/" + filename,  
            "raw_text": text_output,
            "result_json": result_json
        })
    
    except Exception as e:
        return HTMLResponse(content=f"<h1>처리 중 오류 발생: {e}</h1>", status_code=500)


# 루트 페이지: 업로드 폼
@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    # 환경변수에서 서버 URL 읽어오기
    server_url = os.getenv("NGROK_URL", "http://localhost:8000")
    
    return HTMLResponse(f"""
        <html>
        <head><title>신분증 정보 추출</title></head>
        <body>
            <h2>신분증 이미지 업로드</h2>
            <form action="{server_url}/upload" enctype="multipart/form-data" method="post">
              <input name="file" type="file" accept="image/*" required>
              <input type="submit" value="업로드 및 분석">
            </form>
        </body>
        </html>
    """)
