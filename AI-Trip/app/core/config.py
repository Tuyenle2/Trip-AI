import os
import ssl
from dotenv import load_dotenv

def setup_env():
    """Vô hiệu hóa SSL và nạp biến môi trường từ .env"""
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["SSL_CERT_FILE"] = ""
    load_dotenv()