from fastapi import APIRouter, HTTPException
from app.schemas.analysis_schema import DifficultyAnalysisRequest
from app.services.analysis_service import analysis_service

router = APIRouter(prefix="/api/resources", tags=["Difficulty Analysis"])

@router.post("/{resource_id}/analysis")
async def analyze_text_difficulty(resource_id: str, payload: DifficultyAnalysisRequest):
    try:
        if not payload.sentence.strip():
            raise HTTPException(status_code=400, detail="텍스트 내용이 비어있습니다.")

        predicted_score, difficulty_level = analysis_service.predict_score(payload.sentence)

        return {
            "resource_id": resource_id,
            "status": "success",
            "difficulty_score": predicted_score,
            "difficulty_level": difficulty_level
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 중 에러 발생: {str(e)}")