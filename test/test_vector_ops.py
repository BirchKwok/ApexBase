"""
Tests for vector distance functions.

Covers:
- encode_vector / decode_vector round-trip
- array_distance (L2), cosine_similarity, cosine_distance,
  inner_product, l1_distance, linf_distance
- ORDER BY array_distance(...) LIMIT k  (TopK)
- ORDER BY array_distance(col, [...])   (expression ORDER BY without alias)
- Auto-encoding of list/numpy vector columns in store()
- vector_dim, vector_norm, vector_to_string utilities
"""

import math
import struct
import numpy as np
import pytest

from apexbase.client import ApexClient, encode_vector, decode_vector


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path):
    c = ApexClient(dirpath=str(tmp_path), drop_if_exists=True)
    c.create_table("vecs")
    yield c
    c.close()


# ─────────────────────────────────────────────────────────────────────────────
# encode / decode helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_encode_decode_list():
    v = [1.0, 2.0, 3.0]
    assert decode_vector(encode_vector(v)) == pytest.approx(v, abs=1e-6)


def test_encode_decode_numpy():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    decoded = decode_vector(encode_vector(v))
    assert decoded == pytest.approx(v.tolist(), abs=1e-6)


def test_encode_decode_empty():
    assert decode_vector(encode_vector([])) == []


# ─────────────────────────────────────────────────────────────────────────────
# Auto-encoding in store()
# ─────────────────────────────────────────────────────────────────────────────

def test_auto_encode_list_vectors(client):
    rows = [
        {"name": "a", "vec": [1.0, 0.0, 0.0]},
        {"name": "b", "vec": [0.0, 1.0, 0.0]},
        {"name": "c", "vec": [0.0, 0.0, 1.0]},
    ]
    client.store(rows)
    result = client.execute("SELECT name FROM vecs ORDER BY name").to_dict()
    assert [r["name"] for r in result] == ["a", "b", "c"]


def test_auto_encode_numpy_vectors(client):
    rows = [
        {"name": "x", "vec": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
        {"name": "y", "vec": np.array([4.0, 5.0, 6.0], dtype=np.float32)},
    ]
    client.store(rows)
    result = client.execute("SELECT COUNT(*) FROM vecs").to_dict()
    assert result[0]["COUNT(*)"] == 2




# ─────────────────────────────────────────────────────────────────────────────
# Distance functions
# ─────────────────────────────────────────────────────────────────────────────

def _load_3d_vecs(client):
    rows = [
        {"name": "origin",  "vec": [0.0, 0.0, 0.0]},
        {"name": "unit_x",  "vec": [1.0, 0.0, 0.0]},
        {"name": "unit_y",  "vec": [0.0, 1.0, 0.0]},
        {"name": "unit_z",  "vec": [0.0, 0.0, 1.0]},
        {"name": "diag",    "vec": [1.0, 1.0, 1.0]},
    ]
    client.store(rows)


def test_l2_distance(client):
    _load_3d_vecs(client)
    rows = client.execute(
        "SELECT name, array_distance(vec, [1.0, 0.0, 0.0]) AS dist FROM vecs "
        "ORDER BY name"
    ).to_dict()
    dist = {r["name"]: r["dist"] for r in rows}
    assert dist["unit_x"] == pytest.approx(0.0, abs=1e-5)
    assert dist["origin"] == pytest.approx(1.0, abs=1e-5)
    assert dist["unit_y"] == pytest.approx(math.sqrt(2), abs=1e-5)


def test_l2_distance_alias(client):
    _load_3d_vecs(client)
    rows = client.execute(
        "SELECT name, l2_distance(vec, [1.0, 0.0, 0.0]) AS dist FROM vecs "
        "ORDER BY name"
    ).to_dict()
    dist = {r["name"]: r["dist"] for r in rows}
    assert dist["unit_x"] == pytest.approx(0.0, abs=1e-5)


def test_cosine_similarity(client):
    _load_3d_vecs(client)
    rows = client.execute(
        "SELECT name, cosine_similarity(vec, [1.0, 0.0, 0.0]) AS sim FROM vecs "
        "ORDER BY name"
    ).to_dict()
    sim = {r["name"]: r["sim"] for r in rows}
    assert sim["unit_x"] == pytest.approx(1.0, abs=1e-5)
    assert sim["unit_y"] == pytest.approx(0.0, abs=1e-5)
    assert sim["diag"]   == pytest.approx(1.0 / math.sqrt(3), abs=1e-5)


def test_cosine_distance(client):
    _load_3d_vecs(client)
    rows = client.execute(
        "SELECT name, cosine_distance(vec, [1.0, 0.0, 0.0]) AS d FROM vecs "
        "ORDER BY name"
    ).to_dict()
    d = {r["name"]: r["d"] for r in rows}
    assert d["unit_x"] == pytest.approx(0.0, abs=1e-5)
    assert d["unit_y"] == pytest.approx(1.0, abs=1e-5)


def test_inner_product(client):
    _load_3d_vecs(client)
    rows = client.execute(
        "SELECT name, inner_product(vec, [1.0, 2.0, 3.0]) AS ip FROM vecs "
        "ORDER BY name"
    ).to_dict()
    ip = {r["name"]: r["ip"] for r in rows}
    assert ip["origin"] == pytest.approx(0.0, abs=1e-5)
    assert ip["unit_x"] == pytest.approx(1.0, abs=1e-5)
    assert ip["unit_y"] == pytest.approx(2.0, abs=1e-5)
    assert ip["unit_z"] == pytest.approx(3.0, abs=1e-5)
    assert ip["diag"]   == pytest.approx(6.0, abs=1e-5)


def test_l1_distance(client):
    _load_3d_vecs(client)
    rows = client.execute(
        "SELECT name, l1_distance(vec, [1.0, 0.0, 0.0]) AS d FROM vecs "
        "ORDER BY name"
    ).to_dict()
    d = {r["name"]: r["d"] for r in rows}
    assert d["unit_x"] == pytest.approx(0.0, abs=1e-5)
    assert d["origin"] == pytest.approx(1.0, abs=1e-5)
    assert d["diag"]   == pytest.approx(2.0, abs=1e-5)   # |0|+|1|+|1|


def test_linf_distance(client):
    _load_3d_vecs(client)
    rows = client.execute(
        "SELECT name, linf_distance(vec, [0.5, 0.5, 0.5]) AS d FROM vecs "
        "ORDER BY name"
    ).to_dict()
    d = {r["name"]: r["d"] for r in rows}
    assert d["origin"] == pytest.approx(0.5, abs=1e-5)
    assert d["unit_x"] == pytest.approx(0.5, abs=1e-5)
    assert d["diag"]   == pytest.approx(0.5, abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# TopK queries
# ─────────────────────────────────────────────────────────────────────────────

def _load_many(client, n=100, dim=8):
    rng = np.random.default_rng(42)
    rows = [
        {"id": i, "vec": (rng.random(dim).astype(np.float32)).tolist()}
        for i in range(n)
    ]
    client.store(rows)
    return rows


def test_topk_l2_with_alias(client):
    """ORDER BY aliased distance column LIMIT k."""
    rows = _load_many(client, n=50, dim=4)
    query = [0.5, 0.5, 0.5, 0.5]

    result = client.execute(
        f"SELECT id, array_distance(vec, [{','.join(map(str, query))}]) AS dist "
        "FROM vecs ORDER BY dist LIMIT 5"
    ).to_dict()

    assert len(result) == 5
    # Verify ordering
    dists = [r["dist"] for r in result]
    assert dists == sorted(dists)


def test_topk_l2_expression_orderby(client):
    """ORDER BY array_distance(col, [...]) LIMIT k — no alias needed."""
    rows = _load_many(client, n=50, dim=4)
    query = [0.5, 0.5, 0.5, 0.5]

    result = client.execute(
        f"SELECT id FROM vecs "
        f"ORDER BY array_distance(vec, [{','.join(map(str, query))}]) LIMIT 5"
    ).to_dict()

    assert len(result) == 5
    # Verify these are the same top-5 as the aliased version
    result_alias = client.execute(
        f"SELECT id, array_distance(vec, [{','.join(map(str, query))}]) AS dist "
        "FROM vecs ORDER BY dist LIMIT 5"
    ).to_dict()
    ids_expr = {r["id"] for r in result}
    ids_alias = {r["id"] for r in result_alias}
    assert ids_expr == ids_alias


def test_topk_cosine_similarity_desc(client):
    """ORDER BY cosine_similarity(...) DESC LIMIT k."""
    rows = _load_many(client, n=50, dim=4)
    query = [1.0, 0.0, 0.0, 0.0]

    result = client.execute(
        f"SELECT id, cosine_similarity(vec, [{','.join(map(str, query))}]) AS sim "
        "FROM vecs ORDER BY sim DESC LIMIT 5"
    ).to_dict()

    assert len(result) == 5
    sims = [r["sim"] for r in result]
    assert sims == sorted(sims, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def test_vector_dim(client):
    client.store([{"vec": [1.0, 2.0, 3.0]}, {"vec": [4.0, 5.0]}])
    result = client.execute("SELECT vector_dim(vec) AS dim FROM vecs ORDER BY dim").to_dict()
    dims = sorted([r["dim"] for r in result])
    assert dims == [2, 3]


def test_vector_norm(client):
    client.store([{"vec": [3.0, 4.0]}])
    result = client.execute("SELECT vector_norm(vec) AS n FROM vecs").to_dict()
    assert result[0]["n"] == pytest.approx(5.0, abs=1e-5)


def test_vector_to_string(client):
    client.store([{"vec": [1.0, 0.0]}])
    result = client.execute("SELECT vector_to_string(vec) AS s FROM vecs").to_dict()
    assert result[0]["s"].startswith("[")
    assert "1" in result[0]["s"]


# ─────────────────────────────────────────────────────────────────────────────
# String literal query vector
# ─────────────────────────────────────────────────────────────────────────────

def test_string_literal_query(client):
    """array_distance(col, '[1.0,0.0,0.0]') via string literal."""
    _load_3d_vecs(client)
    result = client.execute(
        "SELECT name, array_distance(vec, '[1.0,0.0,0.0]') AS dist FROM vecs "
        "ORDER BY name"
    ).to_dict()
    d = {r["name"]: r["dist"] for r in result}
    assert d["unit_x"] == pytest.approx(0.0, abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# DuckDB-compatible function names
# ─────────────────────────────────────────────────────────────────────────────

def test_duckdb_names(client):
    _load_3d_vecs(client)
    for fn in ["array_distance", "array_cosine_similarity", "array_inner_product",
               "array_l1_distance", "array_cosine_distance"]:
        result = client.execute(
            f"SELECT {fn}(vec, [1.0, 0.0, 0.0]) AS v FROM vecs LIMIT 1"
        ).to_dict()
        assert len(result) == 1
        assert result[0]["v"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# SQL INSERT with vector literals
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_vec_schema(client):
    """Store one row via Python client to establish the binary column schema."""
    client.store([{"id": 0, "vec": encode_vector([0.0, 0.0, 0.0])}])
    client.flush()


def test_sql_insert_array_literal(client):
    """INSERT INTO … VALUES (id, [f1, f2, f3]) array literal."""
    _bootstrap_vec_schema(client)
    client.execute("INSERT INTO vecs (id, vec) VALUES (1, [1.0, 0.0, 0.0])")
    client.execute("INSERT INTO vecs (id, vec) VALUES (2, [0.0, 1.0, 0.0])")
    client.flush()

    result = client.execute(
        "SELECT id, array_distance(vec, [1.0, 0.0, 0.0]) AS dist FROM vecs ORDER BY dist"
    ).to_dict()
    assert result[0]["id"] == 1
    assert result[0]["dist"] == pytest.approx(0.0, abs=1e-5)
    # result[1] is the bootstrapped origin [0,0,0] with dist=1.0
    assert result[1]["dist"] == pytest.approx(1.0, abs=1e-5)
    # result[2] is [0,1,0] with dist=sqrt(2)
    assert result[2]["dist"] == pytest.approx(math.sqrt(2), abs=1e-4)


def test_sql_insert_string_literal(client):
    """INSERT INTO … VALUES (id, '[f1, f2, f3]') string literal auto-coerced."""
    _bootstrap_vec_schema(client)
    client.execute("INSERT INTO vecs (id, vec) VALUES (1, '[1.0, 0.0, 0.0]')")
    client.execute("INSERT INTO vecs (id, vec) VALUES (2, '[0.0, 1.0, 0.0]')")
    client.flush()

    result = client.execute(
        "SELECT id, array_distance(vec, [1.0, 0.0, 0.0]) AS dist FROM vecs ORDER BY dist"
    ).to_dict()
    assert result[0]["id"] == 1
    assert result[0]["dist"] == pytest.approx(0.0, abs=1e-5)


def test_sql_insert_multi_values(client):
    """INSERT INTO … VALUES (…), (…), (…) multi-row in one statement."""
    _bootstrap_vec_schema(client)
    client.execute(
        "INSERT INTO vecs (id, vec) VALUES "
        "(1, [1.0, 0.0, 0.0]), "
        "(2, [0.0, 1.0, 0.0]), "
        "(3, [0.0, 0.0, 1.0])"
    )
    client.flush()

    result = client.execute("SELECT id FROM vecs ORDER BY id").to_dict()
    assert [r["id"] for r in result] == [0, 1, 2, 3]


def test_sql_insert_negative_components(client):
    """INSERT handles negative float components in array literal."""
    _bootstrap_vec_schema(client)
    client.execute("INSERT INTO vecs (id, vec) VALUES (1, [-1.0, -0.5, 0.0])")
    client.flush()

    result = client.execute(
        "SELECT vector_dim(vec) AS d FROM vecs WHERE id = 1"
    ).to_dict()
    assert result[0]["d"] == 3


def test_sql_insert_then_topk(client):
    """TopK query on rows inserted via SQL INSERT."""
    _bootstrap_vec_schema(client)
    client.execute(
        "INSERT INTO vecs (id, vec) VALUES "
        "(1, [1.0, 0.0, 0.0]), "
        "(2, [0.9, 0.1, 0.0]), "
        "(3, [0.0, 0.0, 1.0])"
    )
    client.flush()

    result = client.execute(
        "SELECT id, array_distance(vec, [1.0, 0.0, 0.0]) AS dist "
        "FROM vecs ORDER BY dist LIMIT 2"
    ).to_dict()
    assert len(result) == 2
    ids = [r["id"] for r in result]
    assert 1 in ids
    dists = [r["dist"] for r in result]
    assert dists == sorted(dists)
