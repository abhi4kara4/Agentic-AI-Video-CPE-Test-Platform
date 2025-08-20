from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Video Capture Configuration
    video_capture_base_url: str = "https://app.catsprd.abc.net"
    video_device_id: int = 12
    video_outlet: int = 5
    video_resolution_w: int = 1920
    video_resolution_h: int = 1080
    screenshot_dir: str = "./screenshots"
    video_fps: int = 5
    
    # Device Control Configuration
    device_api_base_url: str = "https://app.catsprd.abc.net"
    device_mac_address: str = "E8:51:9E:D3:99:DB"
    device_auth_token: str = "cbdc109f-91a9-4ce0-8604-75dc22756596"
    device_key_set: str = "PR1_T2"
    device_allocation_category: str = "AUTOMATION"
    
    # Ollama Configuration
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llava:7b"
    ollama_timeout: int = 60
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # Testing Configuration
    test_timeout: int = 300
    test_retry_count: int = 3
    screenshot_on_failure: bool = True
    
    # Development Configuration
    development_mode: bool = True
    require_device_lock: bool = False
    skip_video_capture: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "./logs/platform.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def video_stream_url(self) -> str:
        """Construct the video stream URL"""
        return (
            f"{self.video_capture_base_url}/rack/cats-rack-sn-557.rack.abc.net:443"
            f"/magiq/video/device/{self.video_device_id}/stream"
            f"?outlet={self.video_outlet}"
            f"&resolution_w={self.video_resolution_w}"
            f"&resolution_h={self.video_resolution_h}"
        )


settings = Settings()


# Create necessary directories
os.makedirs(settings.screenshot_dir, exist_ok=True)
if settings.log_file:
    os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)