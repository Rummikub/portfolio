import os
import uuid
from typing import Optional, BinaryIO
from werkzeug.utils import secure_filename
from flask import current_app
from abc import ABC, abstractmethod

class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    async def save(self, file_data: BinaryIO, filename: str, folder: str = "") -> str:
        """Save file and return file path/URL"""
        pass
    
    @abstractmethod
    async def delete(self, file_path: str) -> bool:
        """Delete file and return success status"""
        pass
    
    @abstractmethod
    async def get_url(self, file_path: str) -> str:
        """Get URL for accessing the file"""
        pass

class LocalStorageBackend(StorageBackend):
    """Local file system storage backend"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    async def save(self, file_data: BinaryIO, filename: str, folder: str = "") -> str:
        """Save file to local storage"""
        # Generate unique filename
        file_extension = os.path.splitext(filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        
        # Create folder path
        full_folder_path = os.path.join(self.base_path, folder)
        os.makedirs(full_folder_path, exist_ok=True)
        
        # Save file
        file_path = os.path.join(full_folder_path, unique_filename)
        with open(file_path, 'wb') as f:
            f.write(file_data.read())
        
        # Return relative path
        return os.path.relpath(file_path, self.base_path)
    
    async def delete(self, file_path: str) -> bool:
        """Delete file from local storage"""
        full_path = os.path.join(self.base_path, file_path)
        try:
            if os.path.exists(full_path):
                os.remove(full_path)
                return True
            return False
        except Exception:
            return False
    
    async def get_url(self, file_path: str) -> str:
        """Get URL for local file (returns file path for now)"""
        return f"/uploads/{file_path}"

class S3StorageBackend(StorageBackend):
    """AWS S3 storage backend (stub implementation)"""
    
    def __init__(self, bucket_name: str, region: str = "us-east-1"):
        self.bucket_name = bucket_name
        self.region = region
        # TODO: Initialize boto3 S3 client
        print("S3 Storage Backend initialized (stub)")
    
    async def save(self, file_data: BinaryIO, filename: str, folder: str = "") -> str:
        """Save file to S3 (stub implementation)"""
        # TODO: Implement S3 upload using boto3
        unique_key = f"{folder}/{uuid.uuid4().hex}_{secure_filename(filename)}"
        print(f"Would upload file to S3: {unique_key}")
        return unique_key
    
    async def delete(self, file_path: str) -> bool:
        """Delete file from S3 (stub implementation)"""
        # TODO: Implement S3 delete using boto3
        print(f"Would delete file from S3: {file_path}")
        return True
    
    async def get_url(self, file_path: str) -> str:
        """Get S3 URL (stub implementation)"""
        # TODO: Generate presigned URL or public URL
        return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{file_path}"

class StorageManager:
    """Storage manager to handle different storage backends"""
    
    def __init__(self, backend: StorageBackend):
        self.backend = backend
    
    async def save_file(self, file_data: BinaryIO, filename: str, workspace_id: str) -> str:
        """Save file with workspace scoping"""
        folder = f"workspace_{workspace_id}"
        return await self.backend.save(file_data, filename, folder)
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file"""
        return await self.backend.delete(file_path)
    
    async def get_file_url(self, file_path: str) -> str:
        """Get file URL"""
        return await self.backend.get_url(file_path)
    
    def is_allowed_file(self, filename: str) -> bool:
        """Check if file type is allowed"""
        allowed_extensions = {
            'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx', 
            'xls', 'xlsx', 'ppt', 'pptx', 'csv', 'json'
        }
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in allowed_extensions

def create_storage_manager() -> StorageManager:
    """Factory function to create storage manager"""
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
    
    # For now, use local storage. In production, could switch to S3
    # based on environment variables
    backend = LocalStorageBackend(upload_folder)
    
    # Example of how to use S3 backend:
    # if current_app.config.get('USE_S3'):
    #     backend = S3StorageBackend(
    #         bucket_name=current_app.config['S3_BUCKET'],
    #         region=current_app.config.get('S3_REGION', 'us-east-1')
    #     )
    
    return StorageManager(backend)