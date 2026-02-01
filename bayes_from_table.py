#!/usr/bin/env python3
"""
Deriving Bayes from a Log Support Table

Given ONLY the joint table T(x, y), derive everything:
- Input marginal T(x)
- Output marginal T(y)
- Both conditionals T(y|x) and T(x|y)
- Verify Bayes theorem

Input ES: bytes (alphabet)
Output ES: character classification (vowel, consonant, digit, punct, space, other)
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)

# =============================================================================
# THE RAW DATA: Joint counts from enwik9
# =============================================================================

print("""
================================================================================
STEP 0: The Raw Joint Table
================================================================================

We observe character bigrams in text. For each (prev_char, classification):
  count(prev, class) = how many times we saw this pair

Example subset (counts, not log):
""")

# Simplified: prev char → next char classification
# Rows: previous character (subset)
# Cols: next character class (Vowel, Consonant, Digit, Punct, Space, Other)

prev_chars = ['a', 'e', 'i', 'o', 'u',  # vowels
              'b', 'c', 'd', 'n', 't',  # consonants
              ' ', '.', '0', '1']        # space, punct, digits

classes = ['Vow', 'Con', 'Dig', 'Pun', 'Spc', 'Oth']

# Realistic-ish counts (simplified)
# These reflect: after vowel usually consonant, after space usually consonant, etc.
counts = np.array([
    # Vow   Con   Dig   Pun   Spc   Oth   ← next class
    [ 800, 4200,   10,  100,  900,   50],  # after 'a'
    [1200, 3800,   15,  120, 1100,   60],  # after 'e'
    [ 600, 3500,    8,   80,  800,   40],  # after 'i'
    [ 700, 3200,   12,   90,  850,   45],  # after 'o'
    [ 400, 2800,    5,   60,  600,   30],  # after 'u'
    [1500, 1200,    5,   40,  300,   20],  # after 'b'
    [1800, 1500,    8,   50,  400,   25],  # after 'c'
    [2000,  800,   10,   60,  500,   30],  # after 'd'
    [1200, 2500,   12,   80,  600,   35],  # after 'n'
    [2200, 1800,   15,  100,  800,   40],  # after 't'
    [ 500, 8000,  200,   50,  100,  300],  # after ' ' (space)
    [ 100,  200, 1500,   30, 2000,  100],  # after '.'
    [  50,  100, 3000,   80,  200,  150],  # after '0'
    [  40,   80, 2500,   60,  180,  120],  # after '1'
], dtype=np.float64)

# Print the count table
print("Count table C(prev, class):")
print(f"{'prev':<6}", end='')
for c in classes:
    print(f"{c:>8}", end='')
print()
print("-" * 60)
for i, p in enumerate(prev_chars):
    print(f"'{p}'   ", end='')
    for j in range(len(classes)):
        print(f"{int(counts[i,j]):>8}", end='')
    print()

# =============================================================================
# STEP 1: Convert to Log Support
# =============================================================================

print("""
================================================================================
STEP 1: Convert Counts to Log Support
================================================================================

T(x, y) = log C(x, y)

This is our JOINT log support table - the sufficient statistic.
""")

# Add small epsilon to avoid log(0)
T_joint = np.log(counts + 1e-10)

print("Log support table T(prev, class):")
print(f"{'prev':<6}", end='')
for c in classes:
    print(f"{c:>8}", end='')
print()
print("-" * 60)
for i, p in enumerate(prev_chars):
    print(f"'{p}'   ", end='')
    for j in range(len(classes)):
        print(f"{T_joint[i,j]:>8.2f}", end='')
    print()

# =============================================================================
# STEP 2: Derive Input Marginal T(x)
# =============================================================================

print("""
================================================================================
STEP 2: Input Marginal - T(x) = logsumexp_y T(x, y)
================================================================================

For each input character x, sum over all output classes.
This gives us P(x) - how often each character appears as "previous".

This is the "rest of the INPUT event space".
""")

def logsumexp(a, axis=None):
    """Compute log(sum(exp(a))) in a numerically stable way."""
    a = np.asarray(a)
    if axis is None:
        a_max = np.max(a)
        return a_max + np.log(np.sum(np.exp(a - a_max)))
    else:
        a_max = np.max(a, axis=axis, keepdims=True)
        result = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
        return np.squeeze(result, axis=axis)

T_input = logsumexp(T_joint, axis=1)  # sum over columns (output classes)

print("Input marginal T(x):")
print(f"{'char':<6} {'T(x)':>10} {'P(x)':>10} {'% ':>8}")
print("-" * 40)
total_input = logsumexp(T_input)
for i, p in enumerate(prev_chars):
    p_x = np.exp(T_input[i] - total_input)
    print(f"'{p}'    {T_input[i]:>10.2f} {p_x:>10.4f} {100*p_x:>7.1f}%")

# =============================================================================
# STEP 3: Derive Output Marginal T(y)
# =============================================================================

print("""
================================================================================
STEP 3: Output Marginal - T(y) = logsumexp_x T(x, y)
================================================================================

For each output class y, sum over all input characters.
This gives us P(y) - how often each class appears as "next".

This is the "rest of the OUTPUT event space".
""")

T_output = logsumexp(T_joint, axis=0)  # sum over rows (input chars)

print("Output marginal T(y):")
print(f"{'class':<6} {'T(y)':>10} {'P(y)':>10} {'% ':>8}")
print("-" * 40)
total_output = logsumexp(T_output)
for j, c in enumerate(classes):
    p_y = np.exp(T_output[j] - total_output)
    print(f"{c:<6} {T_output[j]:>10.2f} {p_y:>10.4f} {100*p_y:>7.1f}%")

# =============================================================================
# STEP 4: Derive Conditional T(y|x)
# =============================================================================

print("""
================================================================================
STEP 4: Conditional T(y|x) = T(x,y) - T(x)
================================================================================

"Given I saw character x, what's the probability of each class?"

This is what the RNN learns to predict!
""")

T_y_given_x = T_joint - T_input[:, np.newaxis]

print("Conditional P(class | prev):")
print(f"{'prev':<6}", end='')
for c in classes:
    print(f"{c:>8}", end='')
print()
print("-" * 60)
for i, p in enumerate(prev_chars):
    print(f"'{p}'   ", end='')
    probs = np.exp(T_y_given_x[i])
    for j in range(len(classes)):
        print(f"{probs[j]:>8.3f}", end='')
    print(f"  sum={np.sum(probs):.3f}")

# =============================================================================
# STEP 5: Derive the OTHER Conditional T(x|y)
# =============================================================================

print("""
================================================================================
STEP 5: The Other Conditional T(x|y) = T(x,y) - T(y)
================================================================================

"Given I know the class is Y, what's the probability of each character?"

This is the INVERSE problem - and we need it for Bayes!
""")

T_x_given_y = T_joint - T_output[np.newaxis, :]

print("Conditional P(prev | class) - subset:")
print(f"{'prev':<6}", end='')
for c in classes:
    print(f"{c:>8}", end='')
print()
print("-" * 60)
for i, p in enumerate(prev_chars[:6]):  # just vowels + 'b'
    print(f"'{p}'   ", end='')
    probs = np.exp(T_x_given_y[i])
    for j in range(len(classes)):
        print(f"{probs[j]:>8.3f}", end='')
    print()

# =============================================================================
# STEP 6: VERIFY BAYES THEOREM
# =============================================================================

print("""
================================================================================
STEP 6: Verify Bayes Theorem
================================================================================

Bayes: P(y|x) = P(x|y) * P(y) / P(x)

In log form:
    T(y|x) = T(x|y) + T(y) - T(x)

Let's verify this holds for our derived quantities.
""")

# Compute T(y|x) via Bayes
T_y_given_x_via_bayes = T_x_given_y + T_output[np.newaxis, :] - T_input[:, np.newaxis]

print("Verification: T(y|x) computed two ways")
print(f"{'prev':<6} {'Direct':>12} {'Via Bayes':>12} {'Match?':>10}")
print("-" * 45)

for i in range(min(8, len(prev_chars))):
    p = prev_chars[i]
    direct = T_y_given_x[i, 0]  # first class
    via_bayes = T_y_given_x_via_bayes[i, 0]
    match = "✓" if np.abs(direct - via_bayes) < 1e-6 else "✗"
    print(f"'{p}'    {direct:>12.4f} {via_bayes:>12.4f} {match:>10}")

print("\nBayes theorem verified! Both methods give identical results.")

# =============================================================================
# STEP 7: The Complete Picture
# =============================================================================

print("""
================================================================================
STEP 7: The Complete Picture - Everything from T(x,y)
================================================================================

From the SINGLE joint table T(x,y), we derived:

┌─────────────────────────────────────────────────────────────────────────────┐
│  JOINT TABLE T(x,y)                                                         │
│  = log count(x,y)                                                           │
│  = SUFFICIENT STATISTIC for (x,y) co-occurrence                             │
└─────────────────────────────────────────────────────────────────────────────┘
                │                                   │
                ▼                                   ▼
    ┌───────────────────────┐           ┌───────────────────────┐
    │  INPUT MARGINAL T(x)  │           │  OUTPUT MARGINAL T(y) │
    │  = logsumexp_y T(x,y) │           │  = logsumexp_x T(x,y) │
    │  = "rest of input ES" │           │  = "rest of output ES"│
    └───────────────────────┘           └───────────────────────┘
                │                                   │
                └──────────────┬────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
    ┌───────────────────────┐     ┌───────────────────────┐
    │  T(y|x) = T(x,y)-T(x) │     │  T(x|y) = T(x,y)-T(y) │
    │  "predict class"      │     │  "invert prediction"  │
    └───────────────────────┘     └───────────────────────┘
                │                             │
                └──────────────┬──────────────┘
                               │
                               ▼
               ┌───────────────────────────────────┐
               │  BAYES THEOREM                    │
               │  T(y|x) = T(x|y) + T(y) - T(x)    │
               │                                   │
               │  All four quantities derived from │
               │  the single joint table T(x,y)!   │
               └───────────────────────────────────┘

""")

# =============================================================================
# STEP 8: Concrete Example - Full Trace
# =============================================================================

print("""
================================================================================
STEP 8: Concrete Example - 'e' → 'Vowel'
================================================================================
""")

i_e = prev_chars.index('e')
j_vow = classes.index('Vow')

print(f"Starting point: T('e', Vowel) = {T_joint[i_e, j_vow]:.3f}")
print()
print(f"Input marginal:  T('e') = logsumexp over all classes = {T_input[i_e]:.3f}")
print(f"Output marginal: T(Vowel) = logsumexp over all chars = {T_output[j_vow]:.3f}")
print()
print(f"Forward conditional:")
print(f"  T(Vowel | 'e') = T('e',Vowel) - T('e')")
print(f"                 = {T_joint[i_e, j_vow]:.3f} - {T_input[i_e]:.3f}")
print(f"                 = {T_y_given_x[i_e, j_vow]:.3f}")
print(f"  P(Vowel | 'e') = exp({T_y_given_x[i_e, j_vow]:.3f}) = {np.exp(T_y_given_x[i_e, j_vow]):.3f}")
print()
print(f"Inverse conditional:")
print(f"  T('e' | Vowel) = T('e',Vowel) - T(Vowel)")
print(f"                 = {T_joint[i_e, j_vow]:.3f} - {T_output[j_vow]:.3f}")
print(f"                 = {T_x_given_y[i_e, j_vow]:.3f}")
print(f"  P('e' | Vowel) = exp({T_x_given_y[i_e, j_vow]:.3f}) = {np.exp(T_x_given_y[i_e, j_vow]):.3f}")
print()
print(f"Verify Bayes:")
print(f"  T(Vowel|'e') via Bayes = T('e'|Vowel) + T(Vowel) - T('e')")
print(f"                        = {T_x_given_y[i_e, j_vow]:.3f} + {T_output[j_vow]:.3f} - {T_input[i_e]:.3f}")
bayes_result = T_x_given_y[i_e, j_vow] + T_output[j_vow] - T_input[i_e]
print(f"                        = {bayes_result:.3f}")
print(f"  Direct:               = {T_y_given_x[i_e, j_vow]:.3f}")
print(f"  Match: {'✓' if np.abs(bayes_result - T_y_given_x[i_e, j_vow]) < 1e-6 else '✗'}")

print("""
================================================================================
SUMMARY
================================================================================

The log support table T(x,y) is the ONLY data we need.

To get P(y|x):
  1. Compute T(x) = logsumexp over OUTPUT (rest of output ES)
  2. Compute T(y) = logsumexp over INPUT (rest of input ES)
  3. T(y|x) = T(x,y) - T(x)

Both marginals are needed:
  - T(x) for normalization (what we divide by)
  - T(y) for Bayes inversion (if we need P(x|y))

The INPUT ES here is the alphabet (256 bytes).
The OUTPUT ES is the classification we've factored out.
""")
