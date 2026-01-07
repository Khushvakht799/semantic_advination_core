"""
Semantic Advination Core - система семантического автодополнения команд
"""

__version__ = '1.0.0'

from storage import *
from core import *

__all__ = [
    'Command', 'CommandTrie', 'TrieStorage',
    'Adivinator', 'Validator', 'Orchestrator'
]