# 신분증 정보 추출 프로젝트

이 프로젝트는 FastAPI와 Qwen2-VL 모델을 사용하여 신분증 이미지에서 정보를 추출하는 웹 애플리케이션입니다.<br>
[피피티](https://www.canva.com/design/DAGoEyThJB8/xj6sFnmJitdmiLiRV2Q8NQ/edit?utm_content=DAGoEyThJB8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) <br>
[허깅페이스](https://huggingface.co/)

### 디렉토리 구조 확인

프로젝트 루트에 다음 디렉토리들이 있는지 확인하세요.:
```
genai/
├── static/
│   └── uploads/
└── templates/
```

## 참고사항

- GPU가 필요한 모델이므로 CUDA 환경에서 실행해야 합니다. (저는 코랩을 사용해서 진행했습니다.)

## API 사용법

### 1. 홈페이지 접속
`GET /` - 파일 업로드 폼이 있는 페이지

### 2. 신분증 이미지 업로드
`POST /upload` - 신분증 이미지를 업로드하고 정보를 추출

## 환경변수 설명

- `GOOGLE_API_KEY`: Google Generative AI API 키
- `BASE_DIR`: 프로젝트 기본 디렉토리 경로
- `STATIC_DIR`: 정적 파일 디렉토리명 (기본값: static)
- `UPLOAD_FOLDER`: 업로드 파일 저장 디렉토리명 (기본값: uploads)
- `TEMPLATE_DIR`: 템플릿 디렉토리명 (기본값: templates)
- `MODEL_ID`: 사용할 Hugging Face 모델 ID
- `NGROK_URL`: 외부 접속용 서버 URL
