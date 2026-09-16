import re

def extract_code(text: str) -> str:
    """Removes python code, demarcated by ```python and ```, from LLM output."""
    if not text:
        return ""
    text = re.sub(r"```(?:python)?\s*", "", text)
    text = text.replace("```", "")
    return text.strip()