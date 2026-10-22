"""Tests for the /evaluate endpoint."""

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from spectrojotometer.magnetic_model import MagneticModel
from visualbondweb import api

client = TestClient(api.app)

J_TRUE = np.array([-0.01, 0.004])
TOL = 1.0e-3


def ring_model():
    """Ring of 6 sites, nearest and next-nearest neighbour couplings."""
    n = 6
    pos = [np.array([float(i), 0.0, 0.0]) for i in range(n)]
    lists = [[(i, (i + d) % n, (0,)) for i in range(n)] for d in (1, 2)]
    return MagneticModel(
        pos, [np.array([float(n), 0.0, 0.0])],
        bond_lists=lists, bond_distances=[1.0, 2.0],
    )


def spin_text(model, confs, seed=0, shift=0.0):
    coeffs = model.coefficient_matrix(confs, False)
    rng = np.random.default_rng(seed)
    energs = coeffs.dot(np.r_[J_TRUE, -100.0]) + rng.uniform(-TOL / 2, TOL / 2, len(confs))
    energs = energs + shift * np.where(np.arange(len(confs)) % 2, 1, -1)
    return "\n".join(f"{e:.8f}\t{c}\t# {i}" for i, (e, c) in enumerate(zip(energs, confs)))


ALL_CONFS = [[0] + [int(b) for b in np.binary_repr(k, 5)] for k in range(32)]


@pytest.fixture
def session():
    sid = client.post("/session/new").json()["session_id"]
    api.sessions[sid]["model"] = ring_model()
    return sid


def evaluate(sid, text, **kw):
    body = {"session_id": sid, "configs_text": text, "energy_tolerance": TOL}
    body.update(kw)
    return client.post("/evaluate", json=body)


NAMES = list(ring_model().bonds)


def bars(data):
    return {p["name"]: p for p in data["parameters"]}


def test_default_method_is_exact(session):
    r = evaluate(session, spin_text(api.sessions[session]["model"], ALL_CONFS))
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["method"] == "exact"
    assert data["acceptance_rate"] is None
    assert not data["incompatible"]
    p = bars(data)
    for name, j in zip(NAMES, J_TRUE):
        assert abs(p[name]["value"] - j) <= p[name]["error"]
        assert p[name]["undetermined"] is False


def test_methods_are_ordered(session):
    text = spin_text(api.sessions[session]["model"], ALL_CONFS)
    err = {}
    for method in ("montecarlo", "exact", "ellipse"):
        data = evaluate(session, text, method=method, mc_steps=3000).json()
        assert data["method"] == method
        err[method] = np.array([bars(data)[n]["error"] for n in NAMES])
    assert np.all(err["montecarlo"] <= err["exact"] * (1 + 1e-9))
    assert np.all(err["exact"] <= err["ellipse"] * (1 + 1e-9))


def test_legacy_use_montecarlo_flag(session):
    text = spin_text(api.sessions[session]["model"], ALL_CONFS)
    data = evaluate(session, text, use_montecarlo=True).json()
    assert data["method"] == "montecarlo"
    assert 0 < data["acceptance_rate"] < 1
    assert evaluate(session, text, use_montecarlo=False).json()["method"] == "ellipse"


def test_unknown_method_rejected(session):
    text = spin_text(api.sessions[session]["model"], ALL_CONFS)
    assert evaluate(session, text, method="bogus").status_code == 422


def test_inequalities_lower_upper(session):
    data = evaluate(session, spin_text(api.sessions[session]["model"], ALL_CONFS)).json()
    assert len(data["inequalities"]) == 2
    for iq in data["inequalities"]:
        assert math.isclose(iq["lower"], iq["center"] - iq["half_width"])
        assert math.isclose(iq["upper"], iq["center"] + iq["half_width"])
        assert iq["lower"] < iq["upper"]
        value = float(np.dot(iq["coefficients"], J_TRUE))
        assert iq["lower"] <= value <= iq["upper"]


def test_undetermined_couplings_are_valid_json(session):
    # Repeating the same configurations cannot determine the couplings.
    confs = [ALL_CONFS[0]] * 2 + [ALL_CONFS[1]]
    r = evaluate(session, spin_text(api.sessions[session]["model"], confs))
    assert r.status_code == 200, r.text
    p = bars(r.json())
    assert any(p[n]["undetermined"] for n in NAMES)
    for n in NAMES:
        if p[n]["undetermined"]:
            assert p[n]["value"] is None and p[n]["error"] is None


def test_incompatible_model(session):
    text = spin_text(api.sessions[session]["model"], ALL_CONFS, shift=1.0)
    data = evaluate(session, text).json()
    assert data["incompatible"]
    assert all(bars(data)[n]["incompatible"] for n in NAMES)
