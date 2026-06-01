import html
import os
import json
from google.cloud import texttospeech
from typing import List, Dict, Any
from google import genai
from google.genai import types

from app.core.prompts import TTS_PAUSE_SYSTEM_PROMPT
from app.schemas.speech_schema import AudioBreakdownResponse

SPEED_MAP = {
    5: "slow",  
    4: "85%",   
    3: "90%",   
    2: "95%",  
    1: "100%"  
}

class SpeechService:
    def __init__(self):
        self.tts_client = texttospeech.TextToSpeechClient()
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name="ko-KR-Wavenet-A",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            pitch=-2.0
        )

        self.ai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def _inject_natural_pauses(self, text: str, level: int) -> str:
        if not text:
            return ""
        
        base_pause = 150 + (level * 30)
        contents = f"""
        [지시사항]
        아래 입력 문장을 분석하여 끊어 읽기 처리를 하세요. 
        숨을 쉬어 가야 하는 주요 지점의 기준 가이드라인 쉼 시간은 {base_pause}ms 입니다. 

        [입력 문장]
        "{text}"
        """

        try:
            response = self.ai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=TTS_PAUSE_SYSTEM_PROMPT, 
                    response_mime_type="application/json",
                    response_schema=AudioBreakdownResponse,
                    temperature=0.1, 
                )
            )
            data = json.loads(response.text)
            raw_chunks = data.get("chunks", [])

            normalized_chunks = []

            if isinstance(raw_chunks, dict):
                raw_chunks = [raw_chunks]
            elif isinstance(raw_chunks, str):
                raw_chunks = [{"chunk_text": raw_chunks, "pause_ms": 0}]

            for item in raw_chunks:
                if isinstance(item, str):
                    normalized_chunks.append({"chunk_text": item, "pause_ms": 0})
                elif isinstance(item, dict):
                    normalized_chunks.append(item)
                    
            return normalized_chunks
            
        except Exception as e:
            print(f"Gemini 연동 실패: {str(e)}")
            return [{"chunk_text": text, "pause_ms": 0}]

    def synthesize_adaptive_audio(self, analyzed_sentences: List[Dict[str, Any]]) -> bytes:
        if not analyzed_sentences:
            raise ValueError("음성 합성을 수행할 지문 데이터가 존재하지 않습니다.")

        ssml_text = "<speak>"

        for s in analyzed_sentences:
            level = int(s.get("difficulty_level", 3))
            text = s.get("sentence_text", "")
            speed = SPEED_MAP.get(level, "100%")  

            chunks = self._inject_natural_pauses(text, level)

            ssml_text += f'<s><prosody rate="{speed}">'

            for i, item in enumerate(chunks):
                chunk_text = html.escape(item.get("chunk_text", ""))
                pause_ms = item.get("pause_ms", 0)

                ssml_text += chunk_text
                if pause_ms > 0 and i < len(chunks) - 1:
                    ssml_text += f'<break time="{pause_ms}ms"/>'

            ssml_text += f'</prosody></s><break time="500ms"/>'

        ssml_text += "</speak>"

        try:
            synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
            response = self.tts_client.synthesize_speech(
                input=synthesis_input, 
                voice=self.voice, 
                audio_config=self.audio_config
            )
            return response.audio_content
        except Exception as e:
            print(f"Google TTS API 연동 실패: {str(e)}")
            print(f"[DEBUG] 전송 시도한 SSML 본문 구조:\n{ssml_text}")
            raise ValueError("오디오 합성 중 에러 발생")

speech_service = SpeechService()