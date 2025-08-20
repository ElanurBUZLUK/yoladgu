from abc import ABC, abstractmethod
from typing import Optional, BinaryIO, Dict, Any, List
import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import structlog
from ..core.config import settings, StorageBackend

logger = structlog.get_logger()


class StorageProvider(ABC):
    """Abstract storage provider interface"""
    
    @abstractmethod
    async def upload_file(self, file_path: str, file_content: BinaryIO, content_type: str = None) -> str:
        """Upload a file and return the URL/path"""
        pass
    
    @abstractmethod
    async def download_file(self, file_path: str) -> Optional[BinaryIO]:
        """Download a file and return file-like object"""
        pass
    
    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """Delete a file and return success status"""
        pass
    
    @abstractmethod
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists"""
        pass
    
    @abstractmethod
    async def get_file_url(self, file_path: str, expires_in: int = 3600) -> str:
        """Get public URL for file (with optional expiration for S3)"""
        pass
    
    @abstractmethod
    async def list_files(self, prefix: str = "") -> List[str]:
        """List files with optional prefix"""
        pass


class LocalStorageProvider(StorageProvider):
    """Local file system storage provider"""
    
    def __init__(self, upload_dir: str = None):
        self.upload_dir = upload_dir or settings.upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)
        logger.info("Local storage initialized", upload_dir=self.upload_dir)
    
    async def upload_file(self, file_path: str, file_content: BinaryIO, content_type: str = None) -> str:
        """Upload file to local storage"""
        full_path = os.path.join(self.upload_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'wb') as f:
            f.write(file_content.read())
        
        logger.info("File uploaded to local storage", file_path=file_path, full_path=full_path)
        return file_path
    
    async def download_file(self, file_path: str) -> Optional[BinaryIO]:
        """Download file from local storage"""
        full_path = os.path.join(self.upload_dir, file_path)
        
        if not os.path.exists(full_path):
            logger.warning("File not found in local storage", file_path=file_path)
            return None
        
        return open(full_path, 'rb')
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from local storage"""
        full_path = os.path.join(self.upload_dir, file_path)
        
        try:
            if os.path.exists(full_path):
                os.remove(full_path)
                logger.info("File deleted from local storage", file_path=file_path)
                return True
            return False
        except OSError as e:
            logger.error("Error deleting file from local storage", file_path=file_path, error=str(e))
            return False
    
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists in local storage"""
        full_path = os.path.join(self.upload_dir, file_path)
        return os.path.exists(full_path)
    
    async def get_file_url(self, file_path: str, expires_in: int = 3600) -> str:
        """Get local file URL (for development)"""
        # In development, return a relative path that can be served by FastAPI
        return f"/uploads/{file_path}"
    
    async def list_files(self, prefix: str = "") -> List[str]:
        """List files in local storage with prefix"""
        files = []
        prefix_path = os.path.join(self.upload_dir, prefix)
        
        if os.path.exists(prefix_path):
            for root, dirs, filenames in os.walk(prefix_path):
                for filename in filenames:
                    rel_path = os.path.relpath(os.path.join(root, filename), self.upload_dir)
                    files.append(rel_path)
        
        return files


class S3StorageProvider(StorageProvider):
    """AWS S3 storage provider"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bucket_name = config["bucket_name"]
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=config.get("endpoint_url"),
            aws_access_key_id=config["access_key_id"],
            aws_secret_access_key=config["secret_access_key"],
            region_name=config.get("region", "us-east-1"),
            use_ssl=config.get("use_ssl", True)
        )
        
        logger.info("S3 storage initialized", bucket_name=self.bucket_name, endpoint=config.get("endpoint_url"))
    
    async def upload_file(self, file_path: str, file_content: BinaryIO, content_type: str = None) -> str:
        """Upload file to S3"""
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            self.s3_client.upload_fileobj(
                file_content,
                self.bucket_name,
                file_path,
                ExtraArgs=extra_args
            )
            
            logger.info("File uploaded to S3", file_path=file_path, bucket=self.bucket_name)
            return file_path
            
        except (ClientError, NoCredentialsError) as e:
            logger.error("Error uploading file to S3", file_path=file_path, error=str(e))
            raise
    
    async def download_file(self, file_path: str) -> Optional[BinaryIO]:
        """Download file from S3"""
        try:
            from io import BytesIO
            file_obj = BytesIO()
            
            self.s3_client.download_fileobj(
                self.bucket_name,
                file_path,
                file_obj
            )
            
            file_obj.seek(0)
            return file_obj
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning("File not found in S3", file_path=file_path)
                return None
            logger.error("Error downloading file from S3", file_path=file_path, error=str(e))
            raise
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from S3"""
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=file_path
            )
            
            logger.info("File deleted from S3", file_path=file_path, bucket=self.bucket_name)
            return True
            
        except ClientError as e:
            logger.error("Error deleting file from S3", file_path=file_path, error=str(e))
            return False
    
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists in S3"""
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=file_path
            )
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            logger.error("Error checking file existence in S3", file_path=file_path, error=str(e))
            return False
    
    async def get_file_url(self, file_path: str, expires_in: int = 3600) -> str:
        """Get presigned URL for S3 file"""
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': file_path
                },
                ExpiresIn=expires_in
            )
            return url
        except ClientError as e:
            logger.error("Error generating presigned URL", file_path=file_path, error=str(e))
            raise
    
    async def list_files(self, prefix: str = "") -> List[str]:
        """List files in S3 with prefix"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append(obj['Key'])
            
            return files
            
        except ClientError as e:
            logger.error("Error listing files in S3", prefix=prefix, error=str(e))
            return []


class StorageService:
    """Main storage service that delegates to appropriate provider"""
    
    def __init__(self):
        self.provider = self._initialize_provider()
        logger.info("Storage service initialized", provider_type=type(self.provider).__name__)
    
    def _initialize_provider(self) -> StorageProvider:
        """Initialize storage provider based on configuration"""
        if settings.storage_backend == StorageBackend.S3:
            config = settings.storage_config
            if not all([config.get("access_key_id"), config.get("secret_access_key"), config.get("bucket_name")]):
                logger.warning("S3 configuration incomplete, falling back to local storage")
                return LocalStorageProvider()
            return S3StorageProvider(config)
        else:
            return LocalStorageProvider()
    
    async def upload_file(self, file_path: str, file_content: BinaryIO, content_type: str = None) -> str:
        """Upload file using configured provider"""
        return await self.provider.upload_file(file_path, file_content, content_type)
    
    async def download_file(self, file_path: str) -> Optional[BinaryIO]:
        """Download file using configured provider"""
        return await self.provider.download_file(file_path)
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file using configured provider"""
        return await self.provider.delete_file(file_path)
    
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists using configured provider"""
        return await self.provider.file_exists(file_path)
    
    async def get_file_url(self, file_path: str, expires_in: int = 3600) -> str:
        """Get file URL using configured provider"""
        return await self.provider.get_file_url(file_path, expires_in)
    
    async def list_files(self, prefix: str = "") -> List[str]:
        """List files using configured provider"""
        return await self.provider.list_files(prefix)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check storage provider health"""
        try:
            # Try to list files to test connectivity
            files = await self.list_files()
            return {
                "status": "healthy",
                "provider": type(self.provider).__name__,
                "file_count": len(files)
            }
        except Exception as e:
            logger.error("Storage health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "provider": type(self.provider).__name__,
                "error": str(e)
            }


# Global storage service instance
storage_service = StorageService()
