# utils\similarity.py

import math
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

class StringSimilarity:
    """
    Класс для вычисления схожести строк и команд.
    Использует разные метрики: косинусная схожесть, Левенштейн, Jaccard.
    """

    @staticmethod
    def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Косинусная схожесть между двумя векторами слов.
        """
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum(vec1[x] * vec2[x] for x in intersection)

        sum1 = sum(vec1[x] ** 2 for x in vec1.keys())
        sum2 = sum(vec2[x] ** 2 for x in vec2.keys())
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if denominator == 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Расстояние Левенштейна (редакционное расстояние).
        """
        if len(s1) < len(s2):
            return StringSimilarity.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def normalized_levenshtein(s1: str, s2: str) -> float:
        """
        Нормализованное расстояние Левенштейна (0-1).
        """
        distance = StringSimilarity.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1.0 - (distance / max_len)

    @staticmethod
    def jaccard_similarity(set1: set, set2: set) -> float:
        """
        Коэффициент Жаккара.
        """
        if not set1 and not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union

    @staticmethod
    def word_vectors(text1: str, text2: str, tokenizer: Any = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Создаёт векторы слов для двух текстов.
        Если передан tokenizer, использует его, иначе разбивает по пробелам.
        """
        if tokenizer and hasattr(tokenizer, 'tokenize'):
            words1 = tokenizer.tokenize(text1)
            words2 = tokenizer.tokenize(text2)
        else:
            words1 = text1.lower().split()
            words2 = text2.lower().split()

        vocab = set(words1) | set(words2)
        vec1 = {word: words1.count(word) for word in vocab}
        vec2 = {word: words2.count(word) for word in vocab}
        return vec1, vec2

    @staticmethod
    def combined_similarity(text1: str, text2: str, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Комбинированная схожесть из нескольких метрик.
        weights: словарь с весами метрик (по умолчанию все равны).
        """
        if weights is None:
            weights = {'cosine': 0.4, 'levenshtein': 0.4, 'jaccard': 0.2}

        # Токенизация для некоторых метрик
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())

        # Вычисляем метрики
        vec1, vec2 = StringSimilarity.word_vectors(text1, text2)
        cosine = StringSimilarity.cosine_similarity(vec1, vec2)
        lev = StringSimilarity.normalized_levenshtein(text1, text2)
        jaccard = StringSimilarity.jaccard_similarity(set1, set2)

        # Взвешенная сумма
        total = (weights.get('cosine', 0) * cosine +
                 weights.get('levenshtein', 0) * lev +
                 weights.get('jaccard', 0) * jaccard)

        return total

    @staticmethod
    def find_best_match(query: str, candidates: List[str], threshold: float = 0.5) -> Tuple[Optional[str], float]:
        """
        Находит лучший вариант среди кандидатов по схожести с запросом.
        Возвращает (лучший_кандидат, оценка_схожести).
        """
        best_match = None
        best_score = 0.0

        for candidate in candidates:
            score = StringSimilarity.combined_similarity(query, candidate)
            if score > best_score:
                best_score = score
                best_match = candidate

        if best_score >= threshold:
            return best_match, best_score
        return None, best_score


# Пример использования
if __name__ == "__main__":
    sim = StringSimilarity()

    s1 = "create new project"
    s2 = "create project new"

    print(f"Cosine similarity: {sim.combined_similarity(s1, s2)}")
    print(f"Levenshtein distance: {sim.levenshtein_distance(s1, s2)}")
    print(f"Normalized Levenshtein: {sim.normalized_levenshtein(s1, s2)}")

    candidates = ["create project", "delete project", "start server", "create new project"]
    best, score = sim.find_best_match("creat new proj", candidates)
    print(f"Best match: {best} (score: {score})")