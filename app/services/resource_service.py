import os
import cloudinary
import cloudinary.uploader
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True 
)

class ResourceService:

    @staticmethod
    def upload_image(file_data: bytes, folder: str = "readflow/images") -> str:
        try:
            response = cloudinary.uploader.upload(
                file_data, 
                folder=folder, 
                unique_filename=True, 
                resource_type="image"
            )
            return response.get("secure_url")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"이미지 업로드 실패: {str(e)}")
        
resource_service = ResourceService()