import os
import torch
from transformers import AutoTokenizer
from app.core.config import settings
from app.ml.difficulty_regression_model import KoELECTRADifficultyRegressor

class AnalysisService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.max_len = 128

    def load_model(self):
            print("모델 및 토크나이저 로드 중")

            self.tokenizer = AutoTokenizer.from_pretrained(settings.WEIGHTS_DIR)
            self.model = KoELECTRADifficultyRegressor(model_name=settings.BASE_MODEL)
            
            if os.path.exists(settings.WEIGHTS_PATH):
                self.model.load_state_dict(torch.load(settings.WEIGHTS_PATH, map_location=self.device))
                print("모델 가중치 로드 완료")
            else:
                raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다.")
                
            self.model.to(self.device)
            self.model.eval()

    def calculate_level(self, score: float) -> int:
        if score < 38.0: return 1
        elif score < 46.0: return 2
        elif score < 56.0: return 3
        elif score < 63.0: return 4
        else: return 5

    def predict_score(self, sentence: str) -> tuple[float, int]:
        if not self.model:
             raise RuntimeError("모델이 초기화되지 않았습니다.")
        
        encoding = self.tokenizer(
            sentence,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

            raw_model_output = outputs.squeeze().item()  
            print(f"[Debug] 모델의 예측값 (0~1): {raw_model_output:.4f}")

            raw_score = raw_model_output * 100
            final_score = max(0.0, min(100.0, raw_score))

        level = self.calculate_level(final_score)

        return round(final_score, 2), int(level)

analysis_service = AnalysisService()
