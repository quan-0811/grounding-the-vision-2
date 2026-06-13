# evaluation/__init__.py

"""
Evaluation package.

Modules:
    evaluation.amber_eval
    evaluation.chair
    evaluation.formatters

Do not eagerly import amber_eval or chair here because they may load heavy
dependencies such as spaCy, NLTK, or cached CHAIR objects.
"""

__all__ = [
    "amber_eval",
    "chair",
    "formatters",
]