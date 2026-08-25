"""Clean step handler (plan 2.1 T1b).

Operates from explicit flags via :class:`data.preprocess.steps.CleanStep`.
LazyFrame-in / LazyFrame-out; the only collect is the per-column length
statistics needed by ``LengthTrim.max_sigma``, matching previous behaviour.
"""

import polars as pl

from data.preprocess.steps import CleanStep, LengthTrim, StepContext

EXCLUDE_QUALITY = [
    "keywords:",
    "keywords:query=",
    "http",
    "abstract",
    "abstract advertisement return to issue",
    "paper accepted for publicationarticle views",
    "altimetric-citations",
    "copyright",
    "©reference to this paper",
    "google scholar",
    "an abstract is not available for this content",
    "published by",
    "preview is available",
    "uses cookies to",
    "log in",
    "log in or register",
    "enter your email",
]

EXCLUDE_LANG = [
    # foreign connectives (remember to include space)
    "de ",
    # chinese characters
    # Particles and function words
    "的",
    "了",
    "在",
    "是",
    "和",
    "与",
    "及",
    "或",
    "为",
    "被",
    "有",
    "无",
    "以",
    "对",
    "对于",
    "根据",
    "按",
    "由",
    # Common academic connectives
    "因此",
    "而且",
    "然而",
    "但",
    "但是",
    "同时",
    "并",
    "并且",
    "此外",
    "另外",
    "进一步",
    "总之",
    "综上",
    "可见",
    # Common verbs (often semantically weak in abstracts)
    "表明",
    "显示",
    "证明",
    "说明",
    "指出",
    "认为",
    "发现",
    "提出",
    "方法",
    "研究",
    "分析",
    "讨论",
    "介绍",
    # Common single char words
    "等",
    # Vowels with accents
    "à",
    "á",
    "â",
    "ã",
    "ä",
    "å",
    "æ",
    "è",
    "é",
    "ê",
    "ë",
    "ì",
    "í",
    "î",
    "ï",
    "ò",
    "ó",
    "ô",
    "õ",
    "ö",
    "ø",
    "œ",
    "ù",
    "ú",
    "û",
    "ü",
    "ý",
    "ÿ",
    # Uppercase versions
    "À",
    "Á",
    "Â",
    "Ã",
    "Ä",
    "Å",
    "Æ",
    "È",
    "É",
    "Ê",
    "Ë",
    "Ì",
    "Í",
    "Î",
    "Ï",
    "Ò",
    "Ó",
    "Ô",
    "Õ",
    "Ö",
    "Ø",
    "Œ",
    "Ù",
    "Ú",
    "Û",
    "Ü",
    "Ý",
    # Consonants with diacriticals
    "ç",
    "Ç",
    "ñ",
    "Ñ",
    "ð",
    "Ð",
    "þ",
    "Þ",
    "ß",
]

__all__ = ["EXCLUDE_LANG", "EXCLUDE_QUALITY", "main", "run_clean"]


def _apply_trim(lf: pl.LazyFrame, col: str, trim: LengthTrim) -> pl.LazyFrame:
    """Corrected length-trim semantics.

    Drop rows with ``len_chars < min_chars`` (hard minimum). When
    ``max_sigma`` is set, compute mean/std once per column (single streaming
    collect) and drop rows with ``len > mean + max_sigma * std``. No lower
    statistical bound.
    """
    len_col = f"{col}_len"
    lf = lf.with_columns(pl.col(col).str.len_chars().alias(len_col))
    if trim.min_chars is not None:
        lf = lf.filter(pl.col(len_col) >= trim.min_chars)
    if trim.max_sigma is not None:
        stats = lf.select(pl.col(len_col)).collect(engine="streaming")
        mean_v = stats[len_col].mean()
        std_v = stats[len_col].std()
        if isinstance(mean_v, (int, float)) and isinstance(std_v, (int, float)):
            mean: float = float(mean_v)
            std: float = float(std_v)
            high = mean + (trim.max_sigma * std)
            lf = lf.filter(pl.col(len_col) <= pl.lit(high))
    return lf.drop(len_col)


def run_clean(
    lf: pl.LazyFrame,
    step: CleanStep,
    ctx: StepContext | None = None,  # noqa: ARG001 — uniform handler signature
) -> pl.LazyFrame:
    """Apply a CleanStep to ``lf``. Lazy in, lazy out."""
    del ctx
    col = step.col

    if step.lowercase:
        lf = lf.with_columns(pl.col(col).str.to_lowercase().alias(col))

    if step.drop_quality:
        lf = lf.with_columns(
            pl.when(pl.col(col).str.contains_any(EXCLUDE_QUALITY))
            .then(pl.lit(None, dtype=lf.schema[col]))
            .otherwise(pl.col(col))
            .alias(col)
        )

    if step.lang_policy == "mark":
        lf = lf.with_columns(
            pl.when(pl.col(col).str.contains_any(EXCLUDE_LANG))
            .then(pl.lit("unknown", dtype=lf.schema["language"]))
            .otherwise(pl.col("language"))
            .alias("language")
        )
    elif step.lang_policy == "drop":
        lf = lf.filter(~pl.col(col).str.contains_any(EXCLUDE_LANG))

    if step.trim is not None:
        lf = _apply_trim(lf, col, step.trim)

    if step.require_terminal_period:
        lf = lf.filter(pl.col(col).str.ends_with("."))

    return lf


def main(
    lf: pl.LazyFrame,
    col: str,
    min_len: int,
    level: int = 1,
) -> pl.LazyFrame:
    """Temporary compat shim mapping old integer levels to CleanStep flags.

    Kept only until plan 2.1 T2 rewires pipeline.py and removes it. Mapping:
    level > 1 -> lowercase; level >= 3 -> trim(min_chars=min_len,
    max_sigma=3.0); level > 4 -> require_terminal_period. Semantics differ
    slightly from the old folded floor ``max(mean - 3σ, min_len)``: min_len
    now acts as a hard minimum alongside the sigma cut.
    """
    step = CleanStep(
        col=col,
        lowercase=level > 1,
        trim=(
            LengthTrim(min_chars=min_len, max_sigma=3.0)
            if level >= 3
            else None
        ),
        require_terminal_period=level > 4,
    )
    return run_clean(lf, step)
