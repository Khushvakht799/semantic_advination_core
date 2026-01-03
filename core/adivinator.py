from .models import *
from storage.trie_storage import CommandTrie

class Adivinator:
    def __init__(self, storage: CommandTrie):
        self.storage = storage
    
    def advinate(self, prefix: str, context: Dict = None) -> AdvinationResult:
        # 1. Точные совпадения
        exact = self.storage.search_exact(prefix)
        if exact:
            return AdvinationResult(
                result_type=AdvinationResultType.FOUND,
                suggestions=self._to_suggestions(exact, "exact_match")
            )
        
        # 2. Похожие (если префикс >= 2 символов)
        if len(prefix) >= 2:
            similar = self.storage.search_similar(prefix, threshold=0.3)
            if similar:
                return AdvinationResult(
                    result_type=AdvinationResultType.PARTIAL_FOUND,
                    suggestions=self._to_suggestions(similar, "partial_match"),
                    confidence=max(s["similarity"] for s in similar)
                )
        
        # 3. Ничего не найдено
        return AdvinationResult(
            result_type=AdvinationResultType.NO_MATCH,
            raw_prefix=prefix,
            raw_context=context or {}
        )