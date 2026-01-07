# utils/tokenizer.py

import re
from typing import List, Optional

class CommandTokenizer:
    """
    Токенизатор для разбивки входной строки на токены (слова/операторы).
    Может обрабатывать кавычки, спецсимволы, стоп-слова.
    """

    def __init__(self, stop_words: Optional[List[str]] = None):
        self.stop_words = set(stop_words) if stop_words else set()
        # Регулярка для разбивки по пробелам, кроме текста в кавычках
        self.pattern = re.compile(r'''([^\s"']+|"[^"]*"|'[^']*')''')

    def tokenize(self, text: str, remove_stop_words: bool = True) -> List[str]:
        """
        Разбивает текст на токены.
        """
        if not text or not isinstance(text, str):
            return []

        # Разделяем с учётом кавычек
        raw_tokens = self.pattern.findall(text)
        tokens = []

        for token in raw_tokens:
            # Убираем кавычки, если токен в них заключён
            if (token.startswith('"') and token.endswith('"')) or \
               (token.startswith("'") and token.endswith("'")):
                token = token[1:-1]

            if token:
                tokens.append(token)

        # Убираем стоп-слова
        if remove_stop_words:
            tokens = [t for t in tokens if t.lower() not in self.stop_words]

        return tokens

    def tokenize_with_positions(self, text: str) -> List[dict]:
        """
        Возвращает токены с их позициями в исходном тексте.
        """
        tokens = []
        for match in self.pattern.finditer(text):
            token = match.group()
            start, end = match.span()
            # Убираем кавычки
            if (token.startswith('"') and token.endswith('"')) or \
               (token.startswith("'") and token.endswith("'")):
                token = token[1:-1]
            tokens.append({
                'token': token,
                'start': start,
                'end': end
            })
        return tokens

    def normalize(self, token: str) -> str:
        """
        Нормализует токен (нижний регистр, убирает лишние символы).
        """
        return token.lower().strip()

# Пример использования
if __name__ == "__main__":
    tokenizer = CommandTokenizer(stop_words=["the", "a", "an", "and", "or"])
    sample = 'create "new project" and start server'
    print(tokenizer.tokenize(sample))
    # Вывод: ['create', 'new project', 'start', 'server']