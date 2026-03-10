class SecurityGuard:
    DANGEROUS_KEYWORDS = [
        "ignore previous", "ignore all", "bỏ qua các lệnh",
        "system prompt", "system message", "câu lệnh hệ thống",
        "show prompt", "hiển thị prompt", "bạn được lập trình",
        "quên đi", "reset instructions"
    ]

    @classmethod
    def is_input_safe(cls, user_input: str) -> bool:
        lower_input = user_input.lower()
        for kw in cls.DANGEROUS_KEYWORDS:
            if kw in lower_input:
                return False
        return True