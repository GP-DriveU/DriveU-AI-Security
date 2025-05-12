from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI
import json

# 환경 변수 로드
load_dotenv()
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("OPENAI_API_KEY"),
)

# FastAPI 앱 정의
app = FastAPI()

# 요청 스키마
class SummaryRequest(BaseModel):
    file_id: int
    text: str  # 실전에서는 DB에서 file_id로 text 조회 후 넘기는 방식 가능

# ✅ 문제 생성 요청용 스키마
class QuestionRequest(BaseModel):
    file_ids: list[int]
    texts: list[str]  # 각 파일의 텍스트, 프론트 또는 서버에서 추출 후 전달



# 요약 API


@app.post("/api/ai/summary")
async def summarize(request: SummaryRequest):
    try: 
        prompt = f"""
        다음 텍스트를 핵심 위주로 3문장 이내로 요약해줘:

        {request.text}
        """

        response = client.responses.create(
            model="gpt-4.1",  # gpt-3.5-turbo 가능
            input=[
                {
                    "role": "system", 
                    "content": "당신은 유능한 요약가입니다."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_output_tokens=300
        )

        summary = response.output_text.strip

        return {
            "file_id": request.file_id,
            "summary": summary
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/generate")
async def generate_questions(request: QuestionRequest):
    try:
        # 모든 텍스트를 하나로 연결
        combined_text = "\n\n".join(request.texts)

        prompt = f"""
너는 문제 생성 AI야. 아래 텍스트들을 모두 읽고, 전체 내용을 기반으로 객관식 2문제와 단답형 1문제를 만들어줘. 반드시 다음 JSON 형식으로 출력해:

{{
  "questions": [
    {{
      "type": "multiple_choice",
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "answer": "..."
    }},
    {{
      "type": "short_answer",
      "question": "...",
      "answer": "..."
    }}
  ]
}}

텍스트:
\"\"\"{combined_text}\"\"\"
"""

        response = client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role": "system", 
                    "content": "넌 교육 문제를 만드는 AI야."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.5,
            max_output_tokens=800
        )

        try:
            parsed = json.loads(response.output_text)
            questions = parsed["questions"]
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"JSON 파싱 실패: {str(e)}")

        return {
            "file_ids": request.file_ids,
            "questions": questions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
   
@app.get("/")
async def root():
    return {"message": "Hello, World!"}
