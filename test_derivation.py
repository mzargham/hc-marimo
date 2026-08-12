"""Tests for the axiom-explicit derivation chain (derivation/).

Three kinds of check:
  * per-lemma: each lemma's derive+verify passes in isolation (on the threaded context);
  * full-chain: run.py executes L1->L8 end to end;
  * axiom hygiene: no phantom axioms, inherited axioms are cited, and every load-bearing
    Symbol assumption is licensed by a declared axiom (symproof "load-bearing accounting").
"""

import os
import sys

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from derivation import axioms as AX
from derivation import symbols as S
from derivation.run import CHAIN, run_chain


# ---- Full chain ----
def test_full_chain_runs():
    ctx, used = run_chain(verbose=False)
    assert "rhs" in ctx, "chain did not reach the characteristic system"
    assert used, "no axioms were recorded"


# ---- Per-lemma (each verifies on the fully threaded context) ----
def test_each_lemma_verifies():
    ctx = {}
    for i, lemma in enumerate(CHAIN, start=1):
        lemma.derive(ctx)
        lemma.verify(ctx)  # raises AssertionError on failure
        assert hasattr(lemma, "TITLE") and hasattr(lemma, "AXIOMS_USED"), \
            f"L{i} missing metadata"


# ---- Key symbolic results are what we claim ----
def test_reduction_matches_canonical():
    ctx = {}
    CHAIN[0].derive(ctx)
    assert sp.simplify(ctx["f1"] - S.f1_canonical) == 0
    assert sp.simplify(ctx["f2"] - S.f2_canonical) == 0


def test_switching_and_costate():
    ctx, _ = run_chain(verbose=False)
    x1, x2, p1, p2, phi = S.x1, S.x2, S.p1, S.p2, S.phi
    assert sp.simplify(ctx["sigma"] - (p2 * x1 - p1 * x2)) == 0
    assert sp.simplify(ctx["p1_dot"] - (-phi * p2)) == 0
    assert sp.simplify(ctx["p2_dot"] - (phi * p1)) == 0
    assert ctx["d_norm_sq"] == 0


# ---- Axiom hygiene ----
def test_no_phantom_axioms():
    """Every axiom a lemma claims to use must exist in the manifest."""
    for lemma in CHAIN:
        for name in lemma.AXIOMS_USED:
            assert name in AX.AXIOMS, f"{lemma.__name__} cites undeclared axiom {name}"


def test_inherited_axioms_are_cited():
    """Inherited theorems must carry a citation; that is the whole point of the tag."""
    for a in AX.by_tag(AX.INHERITED):
        assert a.citation.strip(), f"inherited axiom {a.name} has no citation"


def test_load_bearing_symbol_assumptions_are_declared():
    """Any Symbol carrying a strong assumption must be licensed by a declared axiom.

    'Strong' = anything beyond plain real/commutative (e.g. positive). This is the
    load-bearing accounting: an assumption baked into a Symbol is a hidden axiom unless
    it is registered in SYMBOL_ASSUMPTIONS and points at a real axiom.
    """
    registered = {sym for (sym, _assum, _ax) in S.SYMBOL_ASSUMPTIONS.values()}

    # The registered mappings are internally consistent.
    for tag, (sym_name, assumption, axiom_name) in S.SYMBOL_ASSUMPTIONS.items():
        assert axiom_name in AX.AXIOMS, f"{tag}: licenses missing axiom {axiom_name}"
        sym = getattr(S, sym_name)
        assert getattr(sym, f"is_{assumption}") is True, \
            f"{tag}: {sym_name} is not actually {assumption}"

    # No OTHER module symbol smuggles in a strong assumption without being registered.
    strong = ("positive", "negative", "nonnegative", "nonpositive", "integer")
    for name in dir(S):
        obj = getattr(S, name)
        if isinstance(obj, sp.Symbol):
            for assumption in strong:
                if getattr(obj, f"is_{assumption}") is True:
                    assert obj.name in registered, (
                        f"symbol {obj.name} is {assumption} but is not registered in "
                        f"SYMBOL_ASSUMPTIONS with a licensing axiom"
                    )


def test_every_used_axiom_resolves_and_all_tags_valid():
    _, used = run_chain(verbose=False)
    for name in used:
        a = AX.get(name)
        assert a.tag in (AX.POSITED, AX.INHERITED, AX.REGULARITY)
    # The derivation must actually lean on the regularity assumption somewhere.
    assert "R*" in used, "R* (the C^1 assumption that breaks at the star) is never invoked"


if __name__ == "__main__":
    test_full_chain_runs()
    test_each_lemma_verifies()
    test_reduction_matches_canonical()
    test_switching_and_costate()
    test_no_phantom_axioms()
    test_inherited_axioms_are_cited()
    test_load_bearing_symbol_assumptions_are_declared()
    test_every_used_axiom_resolves_and_all_tags_valid()
    print("ALL derivation tests passed")
