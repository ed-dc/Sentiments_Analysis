import re
import string

def remove_urls(text):
    return re.sub(r'http\S+', '', text)

def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

def remove_emojis(text):
    emoji_pattern = re.compile("[\U00010000-\U0010FFFF]", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_emojis(text):
    emoji_pattern = re.compile("[\U00010000-\U0010FFFF]", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_special_chars(text):
    allowed_chars = set(string.ascii_letters + "áéíóúãõàâêôç ")
    return ''.join(c for c in text if c in allowed_chars)



def clean_text(text):
    if not isinstance(text,str):
        return ''
    text = text.lower().strip()
    text = remove_emojis(text)
    text = remove_mentions(text)
    text = remove_urls(text)
    text = remove_special_chars(text)

    return text
