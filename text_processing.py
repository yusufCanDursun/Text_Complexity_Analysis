import re

def turkce_cumle_tokenize(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def turkce_kelime_tokenize(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return words

