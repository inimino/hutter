#!/usr/bin/env python3
"""
Bayesian Pattern Analysis

Working out the math from joint events to conditional probabilities,
and showing how ES normalization redistributes entropy.
"""

import numpy as np

# =============================================================================
# PART 1: From Log Support to Conditional Probability
# =============================================================================

print("""
================================================================================
PART 1: From Log Support to P(vowel | 'e')
================================================================================

We have a single atomic pattern: 'e' → 'vowel'
with log support T(e, vowel) = log count(e, vowel)

This is a SUFFICIENT STATISTIC - nothing more can be said about this joint event.

To get P(vowel | e), we need the marginals:

    P(vowel | e) = P(e, vowel) / P(e)

In log-support:

    T(vowel | e) = T(e, vowel) - T(e)

where T(e) = log Σ_y exp(T(e, y))  over all output events y

The "rest of the event space" on output side:
    T(e) = log[ exp(T(e,vowel)) + exp(T(e,consonant)) + exp(T(e,digit)) + ... ]

Similarly for input side:
    T(vowel) = log Σ_x exp(T(x, vowel))  over all input events x
""")

# Concrete example
T_e_vowel = 20.0    # log support for e → vowel (high)
T_e_cons = 18.0     # log support for e → consonant (lower)
T_e_digit = 5.0     # log support for e → digit (rare)
T_e_other = 15.0    # other

# Marginal for 'e'
T_e = np.log(np.exp(T_e_vowel) + np.exp(T_e_cons) + np.exp(T_e_digit) + np.exp(T_e_other))

# Conditional
T_vowel_given_e = T_e_vowel - T_e
P_vowel_given_e = np.exp(T_vowel_given_e)

print(f"Example:")
print(f"  T(e, vowel) = {T_e_vowel:.1f}")
print(f"  T(e) = {T_e:.2f}  (marginal = logsumexp over outputs)")
print(f"  T(vowel | e) = T(e,vowel) - T(e) = {T_vowel_given_e:.2f}")
print(f"  P(vowel | e) = exp(T_vowel_given_e) = {P_vowel_given_e:.3f}")

# =============================================================================
# PART 2: Pattern Depth via ES Augmentation
# =============================================================================

print("""
================================================================================
PART 2: Pattern Depth Without Growing the NN
================================================================================

Standard RNN at time t:
    input: x_t (one byte)
    h_{t+1} = tanh(W_ih @ x_t + W_hh @ h_t)

Augmented RNN:
    input: (x_t, ES(x_t))  where ES(x_t) = one-hot ES membership
    h_{t+1} = tanh(W_ih @ x_t + W_ES @ ES(x_t) + W_hh @ h_t)

Key insight: ES(x) is DETERMINISTIC given x.
    - is_vowel('e') = 1
    - is_vowel('x') = 0

So the pattern 'vowel' → 'consonant' can now be captured in ONE tick,
even though it spans what would be TWO levels in a tick-tock architecture.

Effective pattern depth increases:
    - Before: W_hh encodes x_{t-1} → x_t patterns
    - After:  W_hh + W_ES encode (x_{t-1}, ES(x_{t-1})) → x_t patterns

This is "free" - no learning needed for ES membership, it's a lookup table.
""")

# =============================================================================
# PART 3: Deterministic Pattern Extraction
# =============================================================================

print("""
================================================================================
PART 3: Deterministic ES Extraction
================================================================================

Define ES membership functions (lookup tables):

    vowel(x)     = 1 if x ∈ {a,e,i,o,u,A,E,I,O,U} else 0
    digit(x)     = 1 if x ∈ {0,1,2,3,4,5,6,7,8,9} else 0
    punct(x)     = 1 if x ∈ {.,!,?,;,:,...} else 0
    whitespace(x)= 1 if x ∈ {space, newline, tab} else 0

Preprocessing pipeline:

    for each input byte x:
        features = [
            one_hot(x),           # 256 dims
            vowel(x),             # 1 dim
            digit(x),             # 1 dim
            punct(x),             # 1 dim
            whitespace(x),        # 1 dim
        ]
        augmented_input = concat(features)  # 260 dims

    feed augmented_input to RNN

The RNN now sees patterns at two granularities simultaneously:
    - Fine: which specific byte?
    - Coarse: which ES?

This is equivalent to running tick-tock but collapsing it into one step.
""")

# =============================================================================
# PART 4: Correlation Structure
# =============================================================================

print("""
================================================================================
PART 4: Correlation Between 'e' and 'vowel'
================================================================================

'e' and 'vowel' are perfectly correlated in one direction:
    - If x = 'e', then vowel(x) = 1  (deterministic)
    - If vowel(x) = 1, then x ∈ {a,e,i,o,u}  (5 possibilities)

Mutual information:
    I(x; vowel) = H(vowel) - H(vowel | x)
                = H(vowel) - 0  (since vowel is determined by x)
                = H(vowel)

For English text:
    P(vowel) ≈ 0.38  (vowels are ~38% of letters)
    H(vowel) ≈ 0.96 bits

So knowing the byte gives us ~1 bit of "free" information about ES membership.

The Bayesian story:
    P(next | e) contains all information for prediction
    P(next | vowel) contains less (coarser prediction)
    P(next | e, vowel) = P(next | e)  because vowel ⊂ information in e

But the KEY point:
    The RNN with just 'e' must LEARN to extract vowel-ness.
    The RNN with (e, vowel) gets this for FREE.
""")

# =============================================================================
# PART 5: ES Normalization and Entropy Redistribution
# =============================================================================

print("""
================================================================================
PART 5: ES Normalization → Uniform Entropy Redistribution
================================================================================

Entropy decomposition:
    H(X) = H(ES(X)) + H(X | ES(X))
         = "coarse" + "fine"

Example:
    H(byte) = H(which ES?) + H(which byte within ES?)
    8 bits  ≈ 2 bits      + 6 bits

When we normalize BY the ES:

    P(x | ES) = P(x) / P(ES)   for x ∈ ES

Within the vowel ES:
    P('e' | vowel) = P('e') / P(vowel)

This redistributes probability to be more uniform WITHIN each ES.

Before normalization:
    P('e') = 0.12  (common)
    P('a') = 0.08
    P('i') = 0.07
    P('o') = 0.07
    P('u') = 0.03

After normalizing by P(vowel) = 0.37:
    P('e' | vowel) = 0.12 / 0.37 = 0.32
    P('a' | vowel) = 0.08 / 0.37 = 0.22
    ...

The "excess entropy" H(X | ES) remains, but is now measured relative to
a uniform prior WITHIN the ES, not globally.

EMBEDDING INTERPRETATION:

    Standard embedding: e_x represents P(next | x)

    ES-normalized embedding:
        e_x represents P(next | x, ES(x))
        = excess predictive power beyond ES membership

    The ES-normalized embedding keeps only the RESIDUAL information
    that 'e' provides beyond just knowing "it's a vowel".
""")

# Concrete calculation
vowels = ['a', 'e', 'i', 'o', 'u']
# Approximate English frequencies
p_vowel = {'a': 0.082, 'e': 0.127, 'i': 0.070, 'o': 0.075, 'u': 0.028}
p_vowel_total = sum(p_vowel.values())

print("Numerical example:")
print(f"\n  Before normalization (raw frequencies):")
for v in vowels:
    print(f"    P('{v}') = {p_vowel[v]:.3f}")
print(f"    P(vowel) = {p_vowel_total:.3f}")

print(f"\n  After ES normalization:")
for v in vowels:
    p_normalized = p_vowel[v] / p_vowel_total
    print(f"    P('{v}' | vowel) = {p_normalized:.3f}")

# Entropy comparison
H_vowel_unnorm = -sum(p * np.log2(p) for p in p_vowel.values() if p > 0)
H_vowel_norm = -sum((p/p_vowel_total) * np.log2(p/p_vowel_total)
                    for p in p_vowel.values() if p > 0)

print(f"\n  Entropy within vowel ES:")
print(f"    Before: H(vowels) = {H_vowel_unnorm:.3f} bits")
print(f"    After:  H(vowels | vowel) = {H_vowel_norm:.3f} bits")
print(f"    Uniform would be: log2(5) = {np.log2(5):.3f} bits")

print("""
================================================================================
SUMMARY: The Bayesian Pattern Story
================================================================================

1. JOINT → CONDITIONAL:
   T(vowel|e) = T(e,vowel) - T(e)
   Need marginals = "rest of event space"

2. PATTERN DEPTH:
   Feed (x, ES(x)) → effectively doubles pattern depth for free

3. DETERMINISTIC EXTRACTION:
   ES membership is a lookup table, no learning needed

4. CORRELATION:
   'e' determines 'vowel', but RNN must learn this
   Augmentation gives it for free

5. ES NORMALIZATION:
   H(X) = H(ES) + H(X|ES)
   Normalizing keeps excess entropy, redistributes uniformly over ES
   Embedding captures residual beyond ES membership

This is the Bayesian decomposition underlying the tick-tock architecture.
""")
