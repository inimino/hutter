#!/usr/bin/env python3
"""Interpret SVD components of bigram matrix."""

import numpy as np

# Read data
with open('enwik9', 'rb') as f:
    data = f.read(10_000_000)

# Build bigram counts
counts = np.zeros((256, 256), dtype=np.float64)
for i in range(len(data) - 1):
    counts[data[i], data[i+1]] += 1

# Compute log-probabilities
row_sums = counts.sum(axis=1, keepdims=True)
P = np.log2((counts + 0.5) / (row_sums + 0.5 * 256))

# SVD
U, S, Vt = np.linalg.svd(P.T)

def byte_label(b):
    if b == 32: return 'SPACE'
    if b == 10: return 'NEWLINE'
    if b == 9: return 'TAB'
    if 97 <= b <= 122: return f"'{chr(b)}'"  # lowercase
    if 65 <= b <= 90: return f"'{chr(b)}'"   # uppercase
    if 48 <= b <= 57: return f"'{chr(b)}'"   # digit
    if 33 <= b <= 126: return f"'{chr(b)}'"  # printable
    if 0xC0 <= b <= 0xDF: return f"UTF8-2B"  # 2-byte UTF-8 lead
    if 0xE0 <= b <= 0xEF: return f"UTF8-3B"  # 3-byte UTF-8 lead
    if 0x80 <= b <= 0xBF: return f"UTF8-cont"  # UTF-8 continuation
    return f"x{b:02X}"

def categorize(indices, vec, n=8):
    """Group top/bottom bytes by category."""
    top = indices[-n:][::-1]
    bot = indices[:n]

    categories = {}
    for i in top:
        cat = byte_label(i)
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((i, vec[i], '+'))
    for i in bot:
        cat = byte_label(i)
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((i, vec[i], '-'))
    return categories

print("=== SVD COMPONENT INTERPRETATION ===\n")
print("Each component separates byte categories.")
print("PREV = what came before (input direction)")
print("NEXT = what comes after (output direction)\n")

for k in range(6):
    print(f"{'='*60}")
    print(f"COMPONENT {k}: S = {S[k]:.1f} ({100*S[k]**2/sum(S**2):.1f}% of variance)")
    print(f"{'='*60}")

    # Analyze input direction (prev byte)
    v = Vt[k, :]
    idx = np.argsort(v)

    print(f"\nPREV byte (V[{k}]):")
    print(f"  High (+): ", end="")
    for i in idx[-6:][::-1]:
        if 32 <= i <= 126:
            print(f"'{chr(i)}' ", end="")
        else:
            print(f"x{i:02X} ", end="")
    print(f"\n  Low  (-): ", end="")
    for i in idx[:6]:
        if 32 <= i <= 126:
            print(f"'{chr(i)}' ", end="")
        else:
            print(f"x{i:02X} ", end="")

    # Analyze output direction (next byte)
    u = U[:, k]
    idx = np.argsort(u)

    print(f"\n\nNEXT byte (U[{k}]):")
    print(f"  High (+): ", end="")
    for i in idx[-6:][::-1]:
        if 32 <= i <= 126:
            print(f"'{chr(i)}' ", end="")
        else:
            print(f"x{i:02X} ", end="")
    print(f"\n  Low  (-): ", end="")
    for i in idx[:6]:
        if 32 <= i <= 126:
            print(f"'{chr(i)}' ", end="")
        else:
            print(f"x{i:02X} ", end="")

    # Interpretation
    print("\n")

# Show as log support for specific pairs
print("\n" + "="*60)
print("RECONSTRUCTION: Rank-k approximation quality")
print("="*60)

for rank in [1, 2, 4, 8, 16, 32, 64]:
    # Reconstruct P^T from top-k components
    Pk = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]

    # Reconstruction error
    err = np.mean((P.T - Pk)**2)

    # Also compute bpc if we used this as predictor
    print(f"Rank {rank:2d}: MSE = {err:.4f}")
