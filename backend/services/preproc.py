import re
import emoji

def emoji_free(text):
    allchars = [c for c in text]
    emoji_list = [c for c in allchars if c in emoji.EMOJI_DATA]
    clean_text = ' '.join([word for word in text.split() if not any(e in word for e in emoji_list)])
    return clean_text

def remove_flags(text):
    flag_pattern = re.compile(r'[\U0001F1E6-\U0001F1FF]{2}')
    return flag_pattern.sub('', text)

def remove_flags_and_keycaps(text):
    flag_pattern = re.compile(r'[\U0001F1E6-\U0001F1FF]{2}')
    # *️⃣, 1️⃣, 2️⃣
    keycap_pattern = re.compile(r'[\*\d#]\uFE0F\u20E3')
    text_no_flags = flag_pattern.sub('', text)
    text_cleaned = keycap_pattern.sub('', text_no_flags)

    return text_cleaned
import re

def clean_text(text: str) -> str:

    if not isinstance(text, str):
        return text  

    text = re.sub(r'[\n\r\t,]', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text


def preprocessing(mess):
    new_mes = emoji_free(mess)
    mes2 = remove_flags(new_mes)
    mes3 = remove_flags_and_keycaps(mes2)
    mes4 = clean_text(mes3)
    return mes4