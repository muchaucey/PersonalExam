"""
模型模块初始化
"""

from .llm_models import create_llm_model, PanGuModel
from .embedding_model import create_embedding_model

__all__ = [
    'create_llm_model',
    'PanGuModel',
    'create_embedding_model',
]