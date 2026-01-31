#!/usr/bin/env python3
"""
P2: W_hh Spectral Radius Test

Prediction: Trained W_hh has spectral radius |λ_max| ≈ 1
Rationale: Information preservation requires eigenvalues near unit circle

If spectral radius < 1: exponential decay (information lost)
If spectral radius > 1: exponential growth (unstable)
If spectral radius ≈ 1: information preserved
"""

import numpy as np

HIDDEN_SIZE = 128

def load_whh(path='model.bin'):
    """Load just W_hh from model."""
    with open(path, 'rb') as f:
        # Skip W_ih and b_h
        f.seek(HIDDEN_SIZE * 256 * 4 + HIDDEN_SIZE * 4)
        W_hh = np.frombuffer(f.read(HIDDEN_SIZE * HIDDEN_SIZE * 4),
                             dtype=np.float32).reshape(HIDDEN_SIZE, HIDDEN_SIZE)
    return W_hh

def analyze_whh(W_hh):
    """Compute eigenvalues and spectral properties."""
    eigenvalues = np.linalg.eigvals(W_hh)

    # Spectral radius = max |λ|
    magnitudes = np.abs(eigenvalues)
    spectral_radius = np.max(magnitudes)

    # How many eigenvalues are near unit circle?
    near_unit = np.sum((magnitudes > 0.9) & (magnitudes < 1.1))
    inside_unit = np.sum(magnitudes < 1.0)

    # Eigenvalue distribution
    return {
        'spectral_radius': spectral_radius,
        'eigenvalues': eigenvalues,
        'magnitudes': magnitudes,
        'near_unit_circle': near_unit,
        'inside_unit_circle': inside_unit,
        'mean_magnitude': np.mean(magnitudes),
        'std_magnitude': np.std(magnitudes)
    }

def main():
    print("Loading W_hh...")
    W_hh = load_whh('model.bin')
    print(f"  Shape: {W_hh.shape}")
    print(f"  Frobenius norm: {np.linalg.norm(W_hh):.3f}")

    print("\nComputing eigenvalues...")
    results = analyze_whh(W_hh)

    print("\n" + "="*50)
    print("P2: W_hh SPECTRAL RADIUS TEST")
    print("="*50)

    sr = results['spectral_radius']
    print(f"\n  Spectral radius |λ_max|: {sr:.4f}")
    print(f"  Prediction: 0.9 < |λ_max| < 1.1")

    if 0.9 < sr < 1.1:
        print(f"  Result: CONFIRMED ✓")
    else:
        print(f"  Result: REFUTED ✗")

    print(f"\n  Eigenvalue statistics:")
    print(f"    Mean |λ|: {results['mean_magnitude']:.4f}")
    print(f"    Std |λ|:  {results['std_magnitude']:.4f}")
    print(f"    Near unit circle (0.9-1.1): {results['near_unit_circle']}/{HIDDEN_SIZE}")
    print(f"    Inside unit circle (<1.0):  {results['inside_unit_circle']}/{HIDDEN_SIZE}")

    # Top 10 eigenvalues by magnitude
    idx = np.argsort(results['magnitudes'])[::-1]
    print(f"\n  Top 10 eigenvalues by magnitude:")
    for i in range(10):
        ev = results['eigenvalues'][idx[i]]
        mag = results['magnitudes'][idx[i]]
        if np.imag(ev) != 0:
            print(f"    {i+1}. {ev.real:+.4f} {ev.imag:+.4f}i  |λ|={mag:.4f}")
        else:
            print(f"    {i+1}. {ev.real:+.4f}            |λ|={mag:.4f}")

    # Interpretation
    print("\n" + "="*50)
    print("INTERPRETATION")
    print("="*50)
    if sr > 1.0:
        print(f"  Spectral radius > 1: Hidden state can grow over time.")
        print(f"  But tanh bounds it to [-1,1], preventing explosion.")
    elif sr < 1.0:
        print(f"  Spectral radius < 1: Linear part decays.")
        print(f"  Memory preserved by nonlinearity + input driving.")
    else:
        print(f"  Spectral radius ≈ 1: Edge of stability.")

    if results['inside_unit_circle'] == HIDDEN_SIZE:
        print(f"  All eigenvalues inside unit circle: stable without tanh.")

    # Save for visualization
    np.savez('spectral_results.npz',
             eigenvalues=results['eigenvalues'],
             magnitudes=results['magnitudes'],
             spectral_radius=sr)
    print(f"\n  Results saved to spectral_results.npz")

if __name__ == '__main__':
    main()
