"""Run the full L1 -> L8 derivation, then print the assumption audit.

    uv run python derivation/run.py

Each lemma derives its step, verifies it as a symbolic identity, and reports the axioms it
invoked. The closing AUDIT is the point of the whole exercise: it separates what we
DERIVED (SymPy-checked) from what we LEAN ON (posited choices, inherited theorems, and the
one regularity assumption that breaks at the star).
"""

import os
import sys

# Make `import derivation.*` work whether run as `python derivation/run.py` or `-m`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivation import axioms as AX
from derivation import (
    lemma_01_reduction,
    lemma_02_hamiltonian,
    lemma_03_switching,
    lemma_04_bangbang,
    lemma_05_evader,
    lemma_06_costate,
    lemma_07_conservation,
    lemma_08_characteristic,
)

CHAIN = [
    lemma_01_reduction,
    lemma_02_hamiltonian,
    lemma_03_switching,
    lemma_04_bangbang,
    lemma_05_evader,
    lemma_06_costate,
    lemma_07_conservation,
    lemma_08_characteristic,
]


def run_chain(verbose=True):
    """Execute the chain, threading a shared context. Returns (ctx, axioms_used)."""
    ctx = {}
    used = []
    for i, lemma in enumerate(CHAIN, start=1):
        lemma.derive(ctx)
        lemma.verify(ctx)
        for name in lemma.AXIOMS_USED:
            if name not in used:
                used.append(name)
        if verbose:
            tag = f"L{i}"
            axset = ", ".join(lemma.AXIOMS_USED) if lemma.AXIOMS_USED else "(algebra only)"
            print(f"  [{tag}] PASS  {lemma.TITLE}")
            print(f"         {lemma.STATEMENT}")
            print(f"         axioms: {axset}")
    return ctx, used


def print_audit(used):
    print()
    print("=" * 72)
    print("ASSUMPTION AUDIT - what the derivation actually leans on")
    print("=" * 72)

    posited = [n for n in used if AX.get(n).tag == AX.POSITED]
    inherited = [n for n in used if AX.get(n).tag == AX.INHERITED]
    regularity = [n for n in used if AX.get(n).tag == AX.REGULARITY]

    print("\nDERIVED here (SymPy-verified): L1-L8 above, each a symbolic identity")
    print("  (reduction, H=p.f+1, switching sigma, bang-bang phi*, evader psi*,")
    print("   costate rotation, ||p||^2 conservation, the characteristic system).")

    print("\nPOSITED (modeling choices we simply declare):")
    for n in sorted(posited):
        print(f"  - {AX.get(n)}")

    print("\nINHERITED (external theorems: USED and CITED, not proven here):")
    for n in sorted(inherited):
        print(f"  - {AX.get(n)}")

    print("\nREGULARITY (load-bearing smoothness):")
    for n in regularity:
        print(f"  - {AX.get(n)}")

    print()
    print("-" * 72)
    print("HEADLINE: the smooth derivation is valid ONLY where V* is C^1 (R*).")
    print("It BREAKS on singular surfaces. The dispersal surface (the star in the talk)")
    print("is exactly where the costate p goes multivalued and phi* = -sign(sigma) flips.")
    print("Everything above is honest away from that boundary; the singular surface is the phenomenon.")
    print("-" * 72)


def main():
    print("=" * 72)
    print("Homicidal Chauffeur - axiom-explicit derivation (L1 -> L8)")
    print("=" * 72)
    print()
    _, used = run_chain(verbose=True)
    print_audit(used)
    print("\nALL LEMMAS PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
