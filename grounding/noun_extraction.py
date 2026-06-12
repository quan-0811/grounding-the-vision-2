"""
Noun extraction and noun-token alignment.

PHG uses this later to decide which generated object mentions need grounding.
"""

from __future__ import annotations

from typing import List, Optional

try:
    import nltk
except ImportError:
    nltk = None


SHELL_NOUNS = {
    "variety", "kind", "type", "sort", "form", "category", "class", "genre",
    "subtype", "subset", "group", "set", "series", "sequence", "suite",
    "lineup", "selection", "array", "collection", "assortment", "mix",
    "blend", "combination", "mixture", "package", "bundle", "batch",
    "bunch", "cluster", "stack", "pile", "heap", "portfolio", "inventory",
    "list", "range", "spectrum", "continuum", "aggregation", "pool", "bucket",
}

GENERIC_BUCKETS = {
    "entity", "entities", "object", "objects", "thing", "things", "item",
    "items", "unit", "units", "component", "components", "element",
    "elements", "material", "materials", "content", "contents", "product",
    "products", "article", "articles", "asset", "assets", "resource",
    "resources", "ingredient", "ingredients", "stuff", "substance",
    "substances", "artifact", "artifacts", "entry", "entries", "record",
    "records", "row", "area", "space", "place", "location", "spot", "section",
    "part",
}

MEASURE_NOUNS = {
    "amount", "number", "quantity", "volume", "mass", "weight", "size",
    "degree", "level", "rate", "proportion", "percentage", "share", "ratio",
    "count", "total", "sum", "average", "mean", "median", "portion", "part",
    "piece", "section", "segment", "subset", "member", "instance", "sample",
    "example", "case", "occurrence", "pair", "couple", "trio", "dozen",
    "hundred", "thousand", "million",
}

IMAGE_DESCRIPTION_NOUNS = {
    "image", "photo", "picture", "scene", "view", "frame", "snapshot",
    "visual", "portrait", "landscape", "depiction", "atmosphere",
    "illustration", "rendering", "capture",
}

DIRECTIONAL_NOUNS = {
    "top", "bottom", "middle", "center", "left", "right", "side", "corner",
    "edge", "border", "margin", "foreground", "background", "midground",
    "front", "back", "rear", "frontside", "backside", "surface", "north",
    "south", "east", "west", "northeast", "northwest", "southeast",
    "southwest", "toward", "towards", "nearby", "near", "around", "across",
    "behind", "above", "below", "under", "over", "beside", "between",
}

OUTLIER_NOUNS = (
    SHELL_NOUNS
    .union(GENERIC_BUCKETS)
    .union(MEASURE_NOUNS)
    .union(IMAGE_DESCRIPTION_NOUNS)
    .union(DIRECTIONAL_NOUNS)
)


def detect_nouns(
    text: str,
    joiner: str = " ",
    drop_outliers: bool = True,
) -> List[str]:
    """
    Extract merged noun phrases using NLTK POS tags.

    Example:
        "A man riding a red bike."
        -> ["man", "red bike"] depending on POS tags.

    Requires:
        nltk punkt / punkt_tab
        nltk averaged_perceptron_tagger / averaged_perceptron_tagger_eng
    """

    if nltk is None:
        raise ImportError("Please install nltk first: pip install nltk")

    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)

    noun_tags = {"NN", "NNS", "NNP", "NNPS"}

    merged = []
    i = 0

    while i < len(tokens):
        if tags[i][1] in noun_tags:
            j = i + 1
            phrase = [tokens[i]]

            while j < len(tokens) and tags[j][1] in noun_tags:
                phrase.append(tokens[j])
                j += 1

            noun = joiner.join(phrase).lower().strip()

            if noun and (not drop_outliers or noun not in OUTLIER_NOUNS):
                merged.append(noun)

            i = j
        else:
            i += 1

    return merged


def find_sublist_start(a: List[int], b: List[int]) -> Optional[int]:
    """
    Find the first start index of sublist b inside list a.
    """

    if not b:
        return None

    n = len(a)
    m = len(b)

    for i in range(n - m + 1):
        if a[i : i + m] == b:
            return i

    return None


def find_noun_token_start(
    tokenizer,
    token_ids: List[int],
    noun: str,
) -> Optional[int]:
    """
    Find where a noun phrase begins inside generated token ids.

    Tries both:
        noun
        " " + noun

    because LLaMA-style tokenizers often encode word-start spaces.
    """

    variants = [
        noun,
        " " + noun,
    ]

    for text in variants:
        noun_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        start = find_sublist_start(
            token_ids,
            noun_ids,
        )

        if start is not None:
            return start

    return None