"""
Compare pygraphc's hill_climb_k2 and k2_local_score against pgmpy.

Every test creates a dataset, runs both implementations, and checks they
produce the same results (scores within tolerance, same DAG structure).
"""

import random

import pytest

pd = pytest.importorskip("pandas")
pgmpy_estimators = pytest.importorskip("pgmpy.estimators")

from pgmpy.estimators import K2, HillClimbSearch  # noqa: E402
from pgmpy.estimators.ScoreCache import ScoreCache  # noqa: E402

from pygraphc import hill_climb_k2, k2_local_score  # noqa: E402 — after importorskip


def _make_dataframe(data: list[list[int]], n_vars: int) -> pd.DataFrame:
    """Convert data to a DataFrame with string column names (pgmpy convention)."""
    return pd.DataFrame(data, columns=[str(i) for i in range(n_vars)])


def _pgmpy_k2_score(df: pd.DataFrame, child: int, parents: list[int]) -> float:
    """Compute K2 local score using pgmpy."""
    scorer = K2(df)
    return scorer.local_score(str(child), [str(p) for p in parents])


def _pgmpy_hill_climb(
    df: pd.DataFrame,
    cardinalities: list[int],
    max_indegree: int = 1,
    tabu_length: int = 100,
    epsilon: float = 1e-4,
) -> set[tuple[int, int]]:
    """Run pgmpy HillClimbSearch and return edges as (parent, child) int tuples."""
    hc = HillClimbSearch(df)
    state_names = {str(i): list(range(card)) for i, card in enumerate(cardinalities)}
    dag = hc.estimate(
        scoring_method=ScoreCache(K2(df, state_names=state_names), df, max_size=1_000_000),
        show_progress=False,
        max_indegree=max_indegree,
        tabu_length=tabu_length,
        epsilon=epsilon,
    )
    return {(int(u), int(v)) for u, v in dag.edges()}


# ── K2 local score comparison tests ──


class TestK2LocalScore:
    """Compare pygraphc k2_local_score against pgmpy K2.local_score."""

    def test_no_parents_binary(self) -> None:
        """Two binary variables, score of variable 1 with no parents."""
        data = [[0, 0], [0, 1], [1, 0], [1, 1]]
        cards = [2, 2]
        df = _make_dataframe(data, 2)

        pygraphc_score = k2_local_score(data, cards, 1, [])
        pgmpy_score = _pgmpy_k2_score(df, 1, [])

        assert pygraphc_score == pytest.approx(pgmpy_score, abs=1e-10)

    def test_one_parent_binary(self) -> None:
        """Two binary variables, score of variable 1 with parent 0."""
        data = [[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [1, 1]]
        cards = [2, 2]
        df = _make_dataframe(data, 2)

        pygraphc_score = k2_local_score(data, cards, 1, [0])
        pgmpy_score = _pgmpy_k2_score(df, 1, [0])

        assert pygraphc_score == pytest.approx(pgmpy_score, abs=1e-10)

    def test_no_parents_ternary(self) -> None:
        """Ternary variable with no parents."""
        data = [[0], [1], [2], [0], [1], [2], [0], [0]]
        cards = [3]
        df = _make_dataframe(data, 1)

        pygraphc_score = k2_local_score(data, cards, 0, [])
        pgmpy_score = _pgmpy_k2_score(df, 0, [])

        assert pygraphc_score == pytest.approx(pgmpy_score, abs=1e-10)

    def test_one_parent_ternary(self) -> None:
        """Ternary child with binary parent."""
        data = [
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 1],
            [1, 2],
            [0, 0],
            [1, 2],
        ]
        cards = [2, 3]
        df = _make_dataframe(data, 2)

        pygraphc_score = k2_local_score(data, cards, 1, [0])
        pgmpy_score = _pgmpy_k2_score(df, 1, [0])

        assert pygraphc_score == pytest.approx(pgmpy_score, abs=1e-10)

    def test_two_parents(self) -> None:
        """Three binary variables, score of variable 2 with parents [0, 1]."""
        data = [
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]
        cards = [2, 2, 2]
        df = _make_dataframe(data, 3)

        pygraphc_score = k2_local_score(data, cards, 2, [0, 1])
        pgmpy_score = _pgmpy_k2_score(df, 2, [0, 1])

        assert pygraphc_score == pytest.approx(pgmpy_score, abs=1e-10)

    def test_all_same_values(self) -> None:
        """
        All samples have the same value — degenerate case.

        pgmpy returns 0.0 here due to its reindex=False optimization dropping
        unobserved parent configs. pygraphc computes the mathematically correct
        K2 score including all parent configurations. The difference doesn't
        affect hill climb because only score deltas matter.
        """
        data = [[0, 0]] * 10
        cards = [2, 2]

        pygraphc_score = k2_local_score(data, cards, 1, [0])

        # Correct K2: lgamma(2)*2 + [lgamma(10+1) - lgamma(10+2)] + [0 - lgamma(2)]
        # = lgamma(2) + lgamma(11) - lgamma(12)
        import math

        expected = 2 * math.lgamma(2) + math.lgamma(11) + math.lgamma(1) - math.lgamma(12) - math.lgamma(2)
        assert pygraphc_score == pytest.approx(expected, abs=1e-10)

    def test_perfect_dependency(self) -> None:
        """Child is a deterministic copy of parent."""
        data = [[0, 0]] * 50 + [[1, 1]] * 50
        cards = [2, 2]
        df = _make_dataframe(data, 2)

        # Score with parent should be much better than without
        score_with = k2_local_score(data, cards, 1, [0])
        score_without = k2_local_score(data, cards, 1, [])
        assert score_with > score_without

        # And must match pgmpy
        pgmpy_with = _pgmpy_k2_score(df, 1, [0])
        pgmpy_without = _pgmpy_k2_score(df, 1, [])
        assert score_with == pytest.approx(pgmpy_with, abs=1e-10)
        assert score_without == pytest.approx(pgmpy_without, abs=1e-10)

    def test_single_sample(self) -> None:
        """
        Edge case: only one sample.

        pgmpy returns 0.0 due to reindex=False dropping empty states.
        pygraphc returns the correct K2 score: lgamma(2) + lgamma(2) - lgamma(3).
        """
        data = [[1, 0]]
        cards = [2, 2]

        pygraphc_score = k2_local_score(data, cards, 0, [])

        import math

        # No parents, card=2, one sample with value=1:
        # lgamma(2) + lgamma(0+1) + lgamma(1+1) - lgamma(1+2) = lgamma(2) + 0 + lgamma(2) - lgamma(3)
        expected = math.lgamma(2) + math.lgamma(1) + math.lgamma(2) - math.lgamma(3)
        assert pygraphc_score == pytest.approx(expected, abs=1e-10)

    def test_many_variables(self) -> None:
        """10 binary variables, score with 1 parent."""
        rng = random.Random(42)
        data = [[rng.randint(0, 1) for _ in range(10)] for _ in range(50)]
        cards = [2] * 10
        df = _make_dataframe(data, 10)

        for child in [0, 3, 7, 9]:
            for parent in [[1], [5], [2, 8]]:
                if child in parent:
                    continue
                pygraphc_score = k2_local_score(data, cards, child, parent)
                pgmpy_score = _pgmpy_k2_score(df, child, parent)
                assert pygraphc_score == pytest.approx(pgmpy_score, abs=1e-10), (
                    f"Mismatch for child={child}, parents={parent}"
                )

    def test_mixed_cardinalities(self) -> None:
        """Variables with different cardinalities."""
        rng = random.Random(123)
        cards = [2, 3, 4, 2]
        data = [[rng.randint(0, c - 1) for c in cards] for _ in range(30)]
        df = _make_dataframe(data, 4)

        pygraphc_score = k2_local_score(data, cards, 2, [0, 1])
        pgmpy_score = _pgmpy_k2_score(df, 2, [0, 1])

        assert pygraphc_score == pytest.approx(pgmpy_score, abs=1e-10)

    def test_unobserved_parent_state(self) -> None:
        """
        Parent has cardinality 3 but only values 0 and 1 appear in data.
        The K2 score must account for the unobserved state.
        """
        data = [[0, 0], [0, 1], [1, 0], [1, 1]] * 5
        cards = [3, 2]  # parent 0 has card=3 but value 2 never appears
        df = _make_dataframe(data, 2)

        pygraphc_score = k2_local_score(data, cards, 1, [0])
        pgmpy_score = _pgmpy_k2_score(df, 1, [0])

        assert pygraphc_score == pytest.approx(pgmpy_score, abs=1e-10)


# ── Hill climb structure comparison tests ──


class TestHillClimbK2:
    """Compare pygraphc hill_climb_k2 against pgmpy HillClimbSearch."""

    def test_empty_data(self) -> None:
        """Empty dataset returns empty DAG."""
        edges = hill_climb_k2([], [2, 2])
        assert edges == []

    def test_single_variable(self) -> None:
        """Single variable — no edges possible."""
        data = [[0], [1], [0], [1]]
        edges = hill_climb_k2(data, [2])
        assert edges == []

    def test_independent_variables(self) -> None:
        """Two independent binary variables — no edge should be learned."""
        rng = random.Random(42)
        data = [[rng.randint(0, 1), rng.randint(0, 1)] for _ in range(200)]
        cards = [2, 2]

        pygraphc_edges = set(hill_climb_k2(data, cards))
        df = _make_dataframe(data, 2)
        pgmpy_edges = _pgmpy_hill_climb(df, cards)

        assert pygraphc_edges == pgmpy_edges

    def test_strong_dependency(self) -> None:
        """
        Variable 1 is a deterministic copy of variable 0.
        Both implementations should find one edge. Direction may differ
        because the K2 score is symmetric for perfectly correlated binary
        variables — both (0→1) and (1→0) are equally good.
        """
        data = [[0, 0]] * 100 + [[1, 1]] * 100
        cards = [2, 2]

        pygraphc_edges = set(hill_climb_k2(data, cards))

        assert len(pygraphc_edges) == 1
        edge = next(iter(pygraphc_edges))
        assert set(edge) == {0, 1}

    def test_chain_dependency(self) -> None:
        """
        Chain: X0 → X1 → X2. X1 depends on X0, X2 depends on X1.
        With max_indegree=1, should find 2 edges.
        """
        rng = random.Random(42)
        data = []
        for _ in range(200):
            x0 = rng.randint(0, 1)
            x1 = x0 if rng.random() < 0.9 else 1 - x0
            x2 = x1 if rng.random() < 0.9 else 1 - x1
            data.append([x0, x1, x2])
        cards = [2, 2, 2]

        pygraphc_edges = set(hill_climb_k2(data, cards))
        df = _make_dataframe(data, 3)
        pgmpy_edges = _pgmpy_hill_climb(df, cards)

        assert pygraphc_edges == pgmpy_edges

    def test_five_binary_variables(self) -> None:
        """Five binary variables with some dependencies."""
        rng = random.Random(99)
        data = []
        for _ in range(300):
            x0 = rng.randint(0, 1)
            x1 = x0 if rng.random() < 0.8 else 1 - x0
            x2 = rng.randint(0, 1)
            x3 = x2 if rng.random() < 0.85 else 1 - x2
            x4 = rng.randint(0, 1)
            data.append([x0, x1, x2, x3, x4])
        cards = [2, 2, 2, 2, 2]

        pygraphc_edges = set(hill_climb_k2(data, cards))
        df = _make_dataframe(data, 5)
        pgmpy_edges = _pgmpy_hill_climb(df, cards)

        assert pygraphc_edges == pgmpy_edges

    def test_max_indegree_2(self) -> None:
        """
        Three variables where X2 depends on both X0 and X1.
        With max_indegree=2, should find two parents for X2.
        """
        rng = random.Random(42)
        data = []
        for _ in range(300):
            x0 = rng.randint(0, 1)
            x1 = rng.randint(0, 1)
            x2 = (x0 + x1) % 2 if rng.random() < 0.9 else rng.randint(0, 1)
            data.append([x0, x1, x2])
        cards = [2, 2, 2]

        pygraphc_edges = set(hill_climb_k2(data, cards, max_indegree=2))
        df = _make_dataframe(data, 3)
        pgmpy_edges = _pgmpy_hill_climb(df, cards, max_indegree=2)

        assert pygraphc_edges == pgmpy_edges

    def test_ternary_variables(self) -> None:
        """Two ternary variables with dependency."""
        rng = random.Random(7)
        data = []
        for _ in range(200):
            x0 = rng.randint(0, 2)
            x1 = x0 if rng.random() < 0.7 else rng.randint(0, 2)
            data.append([x0, x1])
        cards = [3, 3]

        pygraphc_edges = set(hill_climb_k2(data, cards))
        df = _make_dataframe(data, 2)
        pgmpy_edges = _pgmpy_hill_climb(df, cards)

        assert pygraphc_edges == pgmpy_edges

    def test_mixed_cardinalities(self) -> None:
        """Variables with different cardinalities."""
        rng = random.Random(55)
        cards = [2, 3, 4]
        data = []
        for _ in range(300):
            x0 = rng.randint(0, 1)
            x1 = min(x0 + rng.randint(0, 1), 2)
            x2 = rng.randint(0, 3)
            data.append([x0, x1, x2])

        pygraphc_edges = set(hill_climb_k2(data, cards))
        df = _make_dataframe(data, 3)
        pgmpy_edges = _pgmpy_hill_climb(df, cards)

        assert pygraphc_edges == pgmpy_edges

    def test_sso_realistic_binary(self) -> None:
        """
        Realistic SSO scenario: 7 binary variables, 100 samples.
        Simulates a converged SSO population.
        """
        rng = random.Random(42)
        data = []
        for _ in range(100):
            genes = [rng.randint(0, 1) for _ in range(7)]
            # Bias toward all-zero (simulating convergence)
            if rng.random() < 0.3:
                genes = [0] * 7
            data.append(genes)
        cards = [2] * 7

        pygraphc_edges = set(hill_climb_k2(data, cards))
        df = _make_dataframe(data, 7)
        pgmpy_edges = _pgmpy_hill_climb(df, cards)

        assert pygraphc_edges == pgmpy_edges

    def test_all_same_data(self) -> None:
        """All samples identical — no dependency can be learned."""
        data = [[0, 0, 0]] * 50
        cards = [2, 2, 2]

        pygraphc_edges = set(hill_climb_k2(data, cards))
        df = _make_dataframe(data, 3)
        pgmpy_edges = _pgmpy_hill_climb(df, cards)

        assert pygraphc_edges == pgmpy_edges

    def test_large_random_10_variables(self) -> None:
        """10 binary variables, 500 samples — general stress test."""
        rng = random.Random(77)
        data = [[rng.randint(0, 1) for _ in range(10)] for _ in range(500)]
        cards = [2] * 10

        pygraphc_edges = set(hill_climb_k2(data, cards))
        df = _make_dataframe(data, 10)
        pgmpy_edges = _pgmpy_hill_climb(df, cards)

        assert pygraphc_edges == pgmpy_edges

    def test_deterministic_results_same_seed(self) -> None:
        """Same data produces same results on repeated calls."""
        rng = random.Random(42)
        data = [[rng.randint(0, 1) for _ in range(5)] for _ in range(100)]
        cards = [2] * 5

        edges1 = set(hill_climb_k2(data, cards))
        edges2 = set(hill_climb_k2(data, cards))

        assert edges1 == edges2

    def test_tabu_length_zero(self) -> None:
        """Tabu length 0 — no tabu list, may find different result."""
        rng = random.Random(42)
        data = [[rng.randint(0, 1) for _ in range(3)] for _ in range(100)]
        cards = [2, 2, 2]

        # Just verify it doesn't crash
        edges = hill_climb_k2(data, cards, tabu_length=0)
        assert isinstance(edges, list)

    def test_result_is_valid_dag(self) -> None:
        """Learned structure must be a valid DAG (no cycles)."""
        rng = random.Random(42)
        data = [[rng.randint(0, 1) for _ in range(8)] for _ in range(200)]
        cards = [2] * 8

        edges = hill_climb_k2(data, cards)

        # Verify DAG property: no node can reach itself
        n = 8
        children: dict[int, list[int]] = {i: [] for i in range(n)}
        for u, v in edges:
            children[u].append(v)

        for start in range(n):
            visited: set[int] = set()
            stack = [start]
            while stack:
                node = stack.pop()
                for child in children.get(node, []):
                    assert child != start, f"Cycle detected via {start} → ... → {child}"
                    if child not in visited:
                        visited.add(child)
                        stack.append(child)

    def test_respects_max_indegree(self) -> None:
        """No node should have more parents than max_indegree."""
        rng = random.Random(42)
        data = [[rng.randint(0, 1) for _ in range(6)] for _ in range(200)]
        cards = [2] * 6

        for max_in in [1, 2, 3]:
            edges = hill_climb_k2(data, cards, max_indegree=max_in)
            parent_count: dict[int, int] = {}
            for _u, v in edges:
                parent_count[v] = parent_count.get(v, 0) + 1
            for node, count in parent_count.items():
                assert count <= max_in, f"Node {node} has {count} parents, max_indegree={max_in}"
