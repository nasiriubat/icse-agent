"""Statistical summaries used by the reporting layer.

Everything here is seeded and dependency-light so a summary is reproducible bit-for-bit.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class Summary:
    """Descriptive summary of a metric sample."""

    n: int
    mean: float
    median: float
    stdev: float
    p50: float
    p95: float
    minimum: float
    maximum: float
    ci_low: float
    ci_high: float
    method: str = "bootstrap_10000_seeded"

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "n": self.n,
            "mean": self.mean,
            "median": self.median,
            "stdev": self.stdev,
            "p50": self.p50,
            "p95": self.p95,
            "min": self.minimum,
            "max": self.maximum,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_method": self.method,
        }


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile; ``q`` in [0, 1]."""
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 20240101,
) -> tuple[float, float]:
    """Seeded percentile bootstrap CI of the mean. Degenerate samples return the point value."""
    if not values:
        return (math.nan, math.nan)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    if len(set(values)) == 1:
        return (float(values[0]), float(values[0]))
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    alpha = (1 - confidence) / 2
    return (percentile(means, alpha), percentile(means, 1 - alpha))


def summarize(values: Sequence[float], *, seed: int = 20240101) -> Summary:
    if not values:
        nan = math.nan
        return Summary(0, nan, nan, nan, nan, nan, nan, nan, nan, nan, "empty")
    low, high = bootstrap_ci(values, seed=seed)
    return Summary(
        n=len(values),
        mean=float(statistics.fmean(values)),
        median=float(statistics.median(values)),
        stdev=float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        p50=percentile(values, 0.50),
        p95=percentile(values, 0.95),
        minimum=float(min(values)),
        maximum=float(max(values)),
        ci_low=low,
        ci_high=high,
    )


def wilson_interval(
    successes: int, trials: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a proportion. Used for rate metrics."""
    if trials == 0:
        return (math.nan, math.nan)
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-9 else _z_for(confidence)
    phat = successes / trials
    denominator = 1 + z**2 / trials
    center = (phat + z**2 / (2 * trials)) / denominator
    margin = (z / denominator) * math.sqrt(phat * (1 - phat) / trials + z**2 / (4 * trials**2))
    return (max(0.0, center - margin), min(1.0, center + margin))


def paired_difference_interval(
    a: Sequence[bool], b: Sequence[bool], *, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Difference of two paired proportions, ``mean(a) - mean(b)``, with its interval.

    Newcombe's hybrid score interval for paired data (method 10; Stat. Med. 17(22):2635-2650,
    1998). It combines the two Wilson intervals with the correlation between the paired
    outcomes. A test
    that fails to reject says only that the data do not separate the two conditions; this
    interval says how large a difference they still allow.
    """
    if len(a) != len(b):
        raise ValueError("paired_difference_interval requires paired samples of equal length")
    n = len(a)
    if n == 0:
        return (math.nan, math.nan, math.nan)
    both = sum(1 for x, y in zip(a, b, strict=True) if x and y)
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    neither = n - both - only_a - only_b
    p1, p2 = (both + only_a) / n, (both + only_b) / n
    l1, u1 = wilson_interval(both + only_a, n, confidence=confidence)
    l2, u2 = wilson_interval(both + only_b, n, confidence=confidence)
    denominator = (both + only_a) * (neither + only_b) * (both + only_b) * (neither + only_a)
    phi = (both * neither - only_a * only_b) / math.sqrt(denominator) if denominator else 0.0
    difference = p1 - p2
    lower = difference - math.sqrt(
        max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2)
    )
    upper = difference + math.sqrt(
        max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2)
    )
    return (difference, max(-1.0, lower), min(1.0, upper))


def _z_for(confidence: float) -> float:
    """Inverse normal CDF via bisection — avoids a scipy dependency."""
    target = (1 + confidence) / 2
    low, high = 0.0, 10.0
    for _ in range(200):
        mid = (low + high) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta effect size in [-1, 1]. 0 means complete overlap."""
    if not a or not b:
        return math.nan
    greater = sum(1 for x in a for y in b if x > y)
    lesser = sum(1 for x in a for y in b if x < y)
    return (greater - lesser) / (len(a) * len(b))


def effect_size_label(delta: float) -> str:
    """Romano et al. thresholds, reported alongside the raw delta."""
    magnitude = abs(delta)
    if math.isnan(magnitude):
        return "undefined"
    if magnitude < 0.147:
        return "negligible"
    if magnitude < 0.33:
        return "small"
    if magnitude < 0.474:
        return "medium"
    return "large"


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_label: str
    n: int
    note: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "test": self.name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "effect_label": self.effect_label,
            "n": self.n,
            "note": self.note,
        }


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> TestResult:
    """Two-sided Mann-Whitney U with a normal approximation and tie correction.

    Small samples get a warning rather than a fabricated exact p-value.
    """
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return TestResult(
            "mann_whitney_u", math.nan, math.nan, math.nan, "undefined", 0, "empty sample"
        )
    combined = [(v, 0) for v in a] + [(v, 1) for v in b]
    combined.sort(key=lambda item: item[0])
    ranks = _average_ranks([value for value, _ in combined])
    rank_sum_a = sum(rank for rank, (_, group) in zip(ranks, combined, strict=True) if group == 0)
    u1 = rank_sum_a - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mean_u = n1 * n2 / 2
    tie_term = _tie_correction([value for value, _ in combined])
    n = n1 + n2
    variance = (n1 * n2 / 12) * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if variance <= 0:
        p_value = 1.0
        z = 0.0
    else:
        z = (u - mean_u) / math.sqrt(variance)
        p_value = 2 * (1 - _normal_cdf(abs(z)))
    delta = cliffs_delta(a, b)
    warnings = []
    if min(n1, n2) < 5:
        warnings.append("normal approximation is unreliable below n=5 per group")
    return TestResult(
        "mann_whitney_u",
        float(u),
        min(1.0, max(0.0, p_value)),
        delta,
        effect_size_label(delta),
        n,
        f"z={z:.4f}",
        warnings,
    )


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> TestResult:
    """Two-sided Wilcoxon signed-rank test on paired samples (normal approximation)."""
    if len(a) != len(b):
        raise ValueError("wilcoxon_signed_rank requires paired samples of equal length")
    differences = [x - y for x, y in zip(a, b, strict=True) if x != y]
    n = len(differences)
    if n == 0:
        return TestResult("wilcoxon_signed_rank", 0.0, 1.0, 0.0, "negligible", 0, "all pairs tied")
    magnitudes = [abs(d) for d in differences]
    ranks = _average_ranks(sorted(magnitudes))
    rank_by_value: dict[float, float] = {}
    for value, rank in zip(sorted(magnitudes), ranks, strict=True):
        rank_by_value.setdefault(value, rank)
    w_plus = sum(rank_by_value[abs(d)] for d in differences if d > 0)
    w_minus = sum(rank_by_value[abs(d)] for d in differences if d < 0)
    w = min(w_plus, w_minus)
    mean_w = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24
    z = (w - mean_w) / math.sqrt(variance) if variance > 0 else 0.0
    p_value = 2 * (1 - _normal_cdf(abs(z)))
    delta = cliffs_delta(a, b)
    warnings = ["normal approximation is unreliable below n=6 pairs"] if n < 6 else []
    return TestResult(
        "wilcoxon_signed_rank",
        float(w),
        min(1.0, max(0.0, p_value)),
        delta,
        effect_size_label(delta),
        n,
        f"z={z:.4f}",
        warnings,
    )


#: Below this many discordant pairs the chi-square approximation is not trustworthy, so the
#: exact binomial form is used instead. 25 is the conventional cutoff.
EXACT_MCNEMAR_THRESHOLD = 25


def cohens_g_label(g: float) -> str:
    """Cohen's thresholds for ``g``, the departure of a discordant proportion from one half."""
    magnitude = abs(g)
    if math.isnan(magnitude):
        return "undefined"
    if magnitude < 0.05:
        return "negligible"
    if magnitude < 0.15:
        return "small"
    if magnitude < 0.25:
        return "medium"
    return "large"


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> TestResult:
    """Two-sided McNemar test on paired binary outcomes.

    This is the right test for our comparisons and the others here are not: two backends answer
    *the same* query set, so their per-query correct/incorrect vectors are paired, and only the
    queries they disagree on carry information. Comparing the two accuracy rates as if they were
    independent samples would overstate the evidence.

    The exact binomial form is used while the discordant count is small, because the chi-square
    approximation is unreliable there and a fabricated p-value is worse than a slower one. Above
    that threshold we use chi-square with Edwards's continuity correction.

    ``statistic`` is the discordant pair count that favours ``a``; ``effect_size`` is Cohen's
    ``g``, the departure of that count's share from one half.
    """
    if len(a) != len(b):
        raise ValueError("mcnemar requires paired samples of equal length")
    n = len(a)
    if n == 0:
        return TestResult("mcnemar", math.nan, math.nan, math.nan, "undefined", 0, "empty sample")
    # b_count: a right where b was wrong. c_count: the reverse. Concordant pairs are discarded,
    # which is the whole point of the test.
    b_count = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    c_count = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    discordant = b_count + c_count
    if discordant == 0:
        return TestResult(
            "mcnemar",
            0.0,
            1.0,
            0.0,
            "negligible",
            n,
            "the two conditions were correct on exactly the same queries",
        )

    warnings: list[str] = []
    if discordant <= EXACT_MCNEMAR_THRESHOLD:
        # Two-sided exact binomial against p = 1/2.
        smaller = min(b_count, c_count)
        tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) * (0.5**discordant)
        p_value = min(1.0, 2 * tail)
        note = f"exact binomial, b={b_count}, c={c_count}"
    else:
        chi_square = max(0, abs(b_count - c_count) - 1) ** 2 / discordant
        # P(chi-square_1 > x) = erfc(sqrt(x / 2)); erfc keeps precision where 1 - cdf underflows.
        p_value = math.erfc(math.sqrt(chi_square / 2))
        note = f"chi-square with continuity correction, b={b_count}, c={c_count}"
    if discordant < 10:
        warnings.append(f"only {discordant} discordant pairs: this comparison has little power")

    g = b_count / discordant - 0.5
    return TestResult(
        "mcnemar",
        float(b_count),
        min(1.0, max(0.0, p_value)),
        g,
        cohens_g_label(g),
        n,
        note,
        warnings,
    )


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * m
    running = 0.0
    for rank, (original_index, p) in enumerate(indexed):
        candidate = min(1.0, (m - rank) * p)
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted


#: Above this many streams the exact sign-flip enumeration (2^k) stops being cheap.
MAX_EXACT_SIGN_FLIP_STREAMS = 22


@dataclass
class StreamTest:
    """A paired comparison at the level of independent streams (experimental units)."""

    n_streams: int
    mean: float
    ci_low: float
    ci_high: float
    p_value: float
    minimum: float
    maximum: float
    favouring_reference: int
    method: str = "cluster_bootstrap_10000_seeded+exact_sign_flip"


def stream_level_test(
    differences: Sequence[float], *, resamples: int = 10_000, seed: int = 20240101
) -> StreamTest:
    """Paired test across independent streams, for outcomes that are clustered within streams.

    Queries that share one memory store are not independent, so a pooled per-query test can
    overstate the evidence. Here each stream contributes one number, its paired accuracy
    difference, and the inference uses only those: a seeded percentile cluster bootstrap for the
    interval, and an exact two-sided sign-flip test (all ``2^k`` sign assignments under the null
    that each stream's difference is symmetric about zero) for the p-value. With ``k`` streams the
    smallest attainable p is ``2 / 2^k``, which is the honest ceiling on what the design can say.
    """
    k = len(differences)
    if k == 0:
        nan = math.nan
        return StreamTest(0, nan, nan, nan, nan, nan, nan, 0, "empty")
    if k > MAX_EXACT_SIGN_FLIP_STREAMS:
        raise ValueError(f"exact sign-flip test enumerates 2^{k} assignments; too many streams")
    values = [float(d) for d in differences]
    observed = abs(sum(values))
    tolerance = 1e-12
    extreme = 0
    for mask in range(1 << k):
        total = 0.0
        for index, value in enumerate(values):
            total += -value if mask >> index & 1 else value
        if abs(total) >= observed - tolerance:
            extreme += 1
    low, high = bootstrap_ci(values, resamples=resamples, seed=seed)
    return StreamTest(
        n_streams=k,
        mean=float(statistics.fmean(values)),
        ci_low=low,
        ci_high=high,
        p_value=extreme / (1 << k),
        minimum=min(values),
        maximum=max(values),
        favouring_reference=sum(1 for value in values if value > 0),
    )


def _average_ranks(sorted_values: Sequence[float]) -> list[float]:
    ranks = [0.0] * len(sorted_values)
    index = 0
    while index < len(sorted_values):
        end = index
        while end + 1 < len(sorted_values) and sorted_values[end + 1] == sorted_values[index]:
            end += 1
        average = (index + end) / 2 + 1
        for position in range(index, end + 1):
            ranks[position] = average
        index = end + 1
    return ranks


def _tie_correction(sorted_values: Sequence[float]) -> float:
    total = 0.0
    index = 0
    while index < len(sorted_values):
        end = index
        while end + 1 < len(sorted_values) and sorted_values[end + 1] == sorted_values[index]:
            end += 1
        group = end - index + 1
        total += group**3 - group
        index = end + 1
    return total


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
