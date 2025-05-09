from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import openai

# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI 앱 정의
app = FastAPI()

# 요청 스키마


class SummarizeRequest(BaseModel):
    text: str

# 요약 API


@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    try:
        prompt = f"""
        다음 텍스트를 핵심 위주로 3문장 이내로 요약해줘:

        {request.text}
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",  # gpt-3.5-turbo 가능
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )

        summary = response.choices[0].message.content.strip()
        return {"summary": summary}

    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Hello, World!"}
