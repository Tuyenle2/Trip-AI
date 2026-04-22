# app/core/logger.py
import logging
import sys

# Tạo một Formatter tùy chỉnh để thêm màu sắc khi in ra Console
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    # Định dạng: [Thời gian] - [Tên file] - [Cấp độ] - [Nội dung]
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def get_logger(name: str):
    """Hàm khởi tạo logger chuyên nghiệp"""
    logger = logging.getLogger(name)
    
    # Tránh việc add handler nhiều lần nếu import ở nhiều nơi
    if not logger.handlers:
        logger.setLevel(logging.INFO) # Mức độ log mặc định
        
        # Cấu hình in ra Console (Terminal/Render Log)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(CustomFormatter())
        
        logger.addHandler(console_handler)
        
    return logger