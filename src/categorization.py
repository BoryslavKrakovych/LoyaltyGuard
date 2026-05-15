"""
Production-grade product categorization.

Replaces ad-hoc AUTO_CATEGORY_RULES with a 3-tier weighted keyword system
and Word2Vec semantic fallback.

Categorization pipeline (in priority order):
  1) Weighted keyword match — deterministic, interpretable.
     - Tier 1 ('high', weight 2.0):   highly specific words (e.g. 'christmas', 'birdhouse')
     - Tier 2 ('medium', weight 1.0): standard category words
     - Tier 3 ('low', weight 0.5):    generic words that can appear anywhere
     Ties broken by CATEGORY_PRIORITY list (most-specific first).
  2) Word2Vec semantic cluster fallback — uses cluster_products() output.
  3) FALLBACK_CATEGORY = 'General Merchandise' if nothing fires.

Public API:
  - categorize_product(text, w2v_lookup=None) → (category, source, score)
  - categorize_dataframe(products_df, name_col, product_clusters=None) → DataFrame with category, category_source, category_confidence
  - build_w2v_lookup(product_clusters) → {product_name: semantic_cluster_name}
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd


CATEGORY_RULES: dict[str, dict[str, set[str]]] = {
    'Seasonal & Holiday': {
        'high': {'christmas', 'xmas', 'santa', 'snowman', 'reindeer', 'advent',
                 'halloween', 'easter', 'valentine', 'noel', 'snowflake'},
        'medium': {'bauble', 'stocking', 'ornament', 'festive', 'angel', 'fairy', 'tinsel'},
    },
    'Christmas Tree & Greenery': {
        'high': {'mistletoe', 'wreath'},
        'medium': {'garland', 'pine'},
        'low': {'tree'},
    },
    'Candles & Tealights': {
        'high': {'candle', 'tealight', 'votive', 'candlestick', 'candleholder', 'wax'},
        'medium': {'lantern', 'incense'},
        'low': {'holder', 'light'},
    },
    'Mugs & Drinkware': {
        'high': {'mug', 'teapot', 'cup', 'saucer', 'tumbler', 'goblet', 'flask', 'chalice'},
        'medium': {'glass', 'bottle'},
    },
    'Tableware': {
        'high': {'plate', 'bowl', 'dish', 'cakestand', 'platter', 'serving'},
        'medium': {'napkin', 'cutlery', 'fork', 'knife', 'spoon', 'tray', 'placemat'},
    },
    'Kitchen Storage': {
        'high': {'lunchbox', 'lunch', 'recipe', 'pantry'},
        'medium': {'tin', 'jar', 'jug', 'kitchen', 'storage', 'canister'},
    },
    'Tea & Coffee': {
        'high': {'tea', 'coffee', 'espresso', 'cafetiere'},
        'medium': {'kettle', 'caddy'},
    },
    'Textiles & Linen': {
        'high': {'towel', 'blanket', 'cushion', 'pillow', 'doormat', 'bunting',
                 'apron', 'tablecloth', 'duvet'},
        'medium': {'throw', 'rug', 'mat', 'curtain', 'linen', 'fabric', 'cloth', 'cover'},
    },
    'Bags & Carriers': {
        'high': {'handbag', 'shopper', 'backpack', 'purse', 'wallet', 'pouch', 'satchel'},
        'medium': {'bag', 'tote', 'case', 'luggage'},
    },
    'Jewellery & Accessories': {
        'high': {'necklace', 'bracelet', 'earring', 'brooch', 'jewellery', 'jewelry',
                 'pendant'},
        'medium': {'ring', 'scarf', 'hat', 'glove', 'cufflink'},
    },
    'Stationery & Cards': {
        'high': {'card', 'cards', 'notebook', 'journal', 'postcard', 'sticker', 'album',
                 'diary'},
        'medium': {'pen', 'pencil', 'paper', 'ribbon', 'wrap', 'wrapping', 'tag', 'envelope'},
    },
    'Gift & Party': {
        'high': {'balloon', 'birthday', 'celebration', 'confetti'},
        'medium': {'gift', 'present', 'party', 'banner', 'streamer'},
    },
    'Garden & Outdoor': {
        'high': {'birdhouse', 'feeder', 'gardeners', 'watering', 'planter'},
        'medium': {'garden', 'plant', 'flower', 'pot', 'outdoor', 'bucket'},
        'low': {'bird'},
    },
    'Kids & Toys': {
        'high': {'toy', 'doll', 'teddy', 'puzzle', 'rattle'},
        'medium': {'kids', 'child', 'children', 'baby', 'game', 'blocks'},
    },
    'Home Decor': {
        'high': {'frame', 'mirror', 'clock', 'vase', 'statue', 'plaque', 'figurine'},
        'medium': {'sign', 'hanger', 'hook', 'wall', 'decor', 'decoration', 'trinket'},
        'low': {'wooden', 'ceramic', 'porcelain', 'heart', 'photo', 'picture',
                'letters', 'block', 'metal'},
    },
}

CATEGORY_PRIORITY: list[str] = [
    'Seasonal & Holiday',
    'Kids & Toys',
    'Christmas Tree & Greenery',
    'Tea & Coffee',
    'Mugs & Drinkware',
    'Tableware',
    'Kitchen Storage',
    'Candles & Tealights',
    'Textiles & Linen',
    'Bags & Carriers',
    'Jewellery & Accessories',
    'Stationery & Cards',
    'Gift & Party',
    'Garden & Outdoor',
    'Home Decor',
]

WEIGHTS: dict[str, float] = {'high': 2.0, 'medium': 1.0, 'low': 0.5}
FALLBACK_CATEGORY = 'General Merchandise'
_TOKEN_RE = re.compile(r'[a-z0-9]+')


def _tokens(text: object) -> set[str]:
    return set(_TOKEN_RE.findall(str(text).lower()))


def categorize_by_keywords(text: object) -> tuple[str, float]:
    """Returns (category_name, weighted_score). Score 0.0 means no keyword hit."""
    text_tokens = _tokens(text)
    if not text_tokens:
        return FALLBACK_CATEGORY, 0.0

    scores: dict[str, float] = {}
    for category, tiers in CATEGORY_RULES.items():
        score = 0.0
        for tier_name, keywords in tiers.items():
            weight = WEIGHTS[tier_name]
            score += weight * len(text_tokens & keywords)
        if score > 0:
            scores[category] = score

    if not scores:
        return FALLBACK_CATEGORY, 0.0

    max_score = max(scores.values())
    winners = [c for c, s in scores.items() if s == max_score]
    if len(winners) == 1:
        return winners[0], max_score

    for category in CATEGORY_PRIORITY:
        if category in winners:
            return category, max_score
    return winners[0], max_score


def build_w2v_lookup(product_clusters: Optional[pd.DataFrame]) -> dict[str, str]:
    """Build {product_name: semantic_cluster_name} from cluster_products() output."""
    if product_clusters is None or product_clusters.empty:
        return {}
    if 'product_name' not in product_clusters.columns:
        return {}
    if 'semantic_cluster_name' not in product_clusters.columns:
        return {}

    work = product_clusters[['product_name', 'semantic_cluster_name']].dropna().copy()
    work['product_name'] = work['product_name'].astype(str).str.strip()
    return dict(zip(work['product_name'], work['semantic_cluster_name']))


def categorize_product(
    text: object,
    w2v_lookup: Optional[dict[str, str]] = None,
) -> tuple[str, str, float]:
    """Categorize one product. Returns (category, source, confidence).

    source ∈ {'keyword', 'semantic', 'fallback'}.
    """
    category, score = categorize_by_keywords(text)
    if score > 0:
        return category, 'keyword', score

    if w2v_lookup:
        key = str(text).strip()
        semantic = w2v_lookup.get(key)
        if semantic and isinstance(semantic, str) and semantic.strip():
            return semantic, 'semantic', 0.0

    return FALLBACK_CATEGORY, 'fallback', 0.0


def categorize_dataframe(
    products: pd.DataFrame,
    name_col: str = 'product_name',
    product_clusters: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add category / category_source / category_confidence columns to products."""
    out = products.copy()
    if name_col not in out.columns:
        out['category'] = FALLBACK_CATEGORY
        out['category_source'] = 'fallback'
        out['category_confidence'] = 0.0
        return out

    w2v_lookup = build_w2v_lookup(product_clusters)

    results = out[name_col].fillna('').apply(
        lambda text: categorize_product(text, w2v_lookup)
    )
    out['category'] = [r[0] for r in results]
    out['category_source'] = [r[1] for r in results]
    out['category_confidence'] = [r[2] for r in results]
    return out
