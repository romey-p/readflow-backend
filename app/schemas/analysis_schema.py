from pydantic import BaseModel, Field

class DifficultyAnalysisRequest(BaseModel):
    sentence: str = Field(..., description="난이도를 분석할 문장", min_length=5)