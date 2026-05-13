import os
import ssl
from dotenv import load_dotenv

def setup_env():
    """Disable SSL and load environment variables from .env"""
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["SSL_CERT_FILE"] = ""
    load_dotenv()