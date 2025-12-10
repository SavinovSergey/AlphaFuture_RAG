import pandas as pd
import re
from typing import List, Tuple, Callable
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np


def remove_common_prefixes(strings):
    """Удаляет общие префиксы из отсортированного списка строк."""
    
    if not strings:
        return []
        
    result = []
    prev = strings[0]
    com_len = 0
    for i, curr in enumerate(strings[1:]):
        # Сохраняем у первого интента общий префикс
        result.append(prev[com_len:])
        # Находим длину общего префикса
        max_com_len = min(len(prev), len(curr))
        while com_len < max_com_len and prev[:com_len] == curr[:com_len]:
            com_len += 1
        # Уменьшаем длину общего префикса, пока не найдем новый общий префикс
        while com_len > 0 and prev[:com_len] != curr[:com_len]:
            com_len -= 1
        prev = curr

    result.append(prev[com_len:])
    return result


def remove_common_prefixes_and_suffixes(strings):
    new_strings = remove_common_prefixes(strings)
    reversed_strings = np.array([s[::-1] for s in new_strings])
    new_strings = remove_common_prefixes(reversed_strings.tolist())
    filtered_strings = [s[::-1] for s in new_strings]
    return filtered_strings


def preprocess_text(text: str) -> str:
    """
    Предобработка текста:
    - Удаление email-адресов
    - Удаление номеров телефонов
    - Удаление специальных символов
    - Замена множественных переносов строк
    """
    # Удаление email и сайтов
    text = re.sub(r'\S+@\S+|(https?:)?www\S+|\S+.(ru|com)', '', text)
    
    # Удаление номеров телефонов
    phone_patterns = [
        r'(\+?[78])?\s*[\(\-]?\s*\d{3}\s*[\)\-]?\s*\d{3}\s*[\-]?\s*\d{2}[\-]?\s*\d{2}',
        r'(\+?[78])?\s*\d{3}\s*[\-]?\s*\d{3}\s*[\-]?\s*\d{2}\s*[\-]?\s*\d{2}',
        r'(\+?[78])?\s*\(\d{3}\)\s*\d{3}[\-]?\d{2}[\-]?\d{2}'
    ]
    for pattern in phone_patterns:
        text = re.sub(pattern, '', text)
    
    # Удаление специальных символов (сохраняем буквы, цифры, пробелы и основную пунктуацию)
    text = re.sub(r'[^\w\s\.\,\!\?\:\;\(\)\-\n\r]', '', text)
    text = re.sub(r'\-{2,}', '', text)
    text = re.sub(r'\.{3,}', '...', text)
    
    # Замена множественных переносов строк на двойной перенос
    text = re.sub(r'[\n\s*]{3,}', '\n\n', text)
    # Замена множественных номеров строк и двойных переносов
    text = re.sub(r'\n\n(\d+\n\n){2,}', '\n\n', text)
    
    # Удаление лишних пробелов
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n ', '\n', text)
    
    return text.strip()


def divide_text(df, len_function):
    split_tags = ["\n\n", "\n", ".", " ", ""]
    divided = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=120,
        separators=split_tags,
        keep_separator="start",
        length_function=len_function
    )
    for index, row in df.iterrows():
        result = splitter.split_text(row['text'])     
        for r in result:
            divided.append([row['web_id'], r])
    return divided