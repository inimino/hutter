#!/usr/bin/env python3
"""
Measure RNN memory depth v2: Conditional mutual information approach.

Instead of perturbing, measure how much the prediction distribution
varies with the value at distance k, controlling for closer context.

If the RNN remembers position -k, then P(next | context) should vary
depending on what byte was at position -k.
"""

import numpy as np
from collections import defaultdict

INPUT_SIZE = 256
HIDDEN_SIZE = 128
OUTPUT_SIZE = 256

def load_model(path='model.bin'):
    with open(path, 'rb') as f:
        W_ih = np.frombuffer(f.read(HIDDEN_SIZE * INPUT_SIZE * 4), dtype=np.float32).reshape(HIDDEN_SIZE, INPUT_SIZE)
        b_h = np.frombuffer(f.read(HIDDEN_SIZE * 4), dtype=np.float32)
        W_hh = np.frombuffer(f.read(HIDDEN_SIZE * HIDDEN_SIZE * 4), dtype=np.float32).reshape(HIDDEN_SIZE, HIDDEN_SIZE)
        W_ho = np.frombuffer(f.read(OUTPUT_SIZE * HIDDEN_SIZE * 4), dtype=np.float32).reshape(OUTPUT_SIZE, HIDDEN_SIZE)
        b_o = np.frombuffer(f.read(OUTPUT_SIZE * 4), dtype=np.float32)
    return {'W_ih': W_ih, 'W_hh': W_hh, 'b_h': b_h, 'W_ho': W_ho, 'b_o': b_o}

def forward_sequence(model, seq):
    h = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    for byte in seq:
        x = np.zeros(INPUT_SIZE, dtype=np.float32)
        x[byte] = 1.0
        h = np.tanh(model['W_ih'] @ x + model['W_hh'] @ h + model['b_h'])
    logits = model['W_ho'] @ h + model['b_o']
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return h, probs

def entropy(probs):
    """Entropy in bits."""
    eps = 1e-10
    return -np.sum(probs * np.log2(probs + eps))

def measure_mi_at_distance(model, data, k, n_samples=1000):
    """
    Estimate I(next; byte_{-k} | bytes_{-k+1}...byte_{-1})

    Group samples by the byte at position -k, compute average prediction
    for each group, measure variance across groups.
    """
    context_len = 50

    # Group predictions by the byte at position -k
    predictions_by_byte = defaultdict(list)

    np.random.seed(42 + k)
    positions = np.random.randint(context_len + 10, len(data) - 1, size=n_samples)

    for pos in positions:
        context = list(data[pos - context_len:pos])
        byte_at_k = context[-k]

        _, probs = forward_sequence(model, context)
        predictions_by_byte[byte_at_k].append(probs)

    # Compute variance of mean predictions across groups
    # Higher variance = more influence of byte at position -k
    group_means = []
    for byte_val, prob_list in predictions_by_byte.items():
        if len(prob_list) >= 5:  # Need enough samples
            mean_prob = np.mean(prob_list, axis=0)
            group_means.append(mean_prob)

    if len(group_means) < 2:
        return 0.0

    group_means = np.array(group_means)

    # Variance of the mean prediction across groups
    # This approximates how much the prediction depends on byte_at_k
    overall_mean = np.mean(group_means, axis=0)
    variance = np.mean(np.sum((group_means - overall_mean) ** 2, axis=1))

    return variance

def main():
    print("Loading model...")
    model = load_model('model.bin')

    print("Loading data...")
    with open('enwik9', 'rb') as f:
        data = f.read(2_000_000)

    print("Measuring conditional dependency at each distance...")
    max_dist = 30
    variances = []

    for k in range(1, max_dist + 1):
        print(f"  Distance {k}...", end=" ", flush=True)
        var = measure_mi_at_distance(model, data, k, n_samples=2000)
        variances.append(var)
        print(f"{var:.6f}")

    # Normalize
    max_var = max(variances)
    if max_var > 0:
        normalized = [v / max_var for v in variances]
    else:
        normalized = variances

    print("\n=== MEMORY DEPTH RESULTS (v2) ===")
    print("Distance k | Variance | Normalized")
    print("-" * 50)

    for k in range(1, max_dist + 1):
        bar_len = int(40 * normalized[k-1])
        bar = '█' * bar_len
        print(f"  {k:2d}       | {variances[k-1]:.6f} | {bar}")

    # Find memory depth: where does it drop to 1/e of max?
    threshold = max(variances) / np.e
    memory_depth = max_dist
    for k in range(1, max_dist + 1):
        if variances[k-1] < threshold:
            memory_depth = k
            break

    print(f"\n  Memory depth (1/e threshold): ~{memory_depth} characters")
    print(f"  Predicted: ~12 characters")

    generate_viz(variances, normalized, max_dist, memory_depth)

def generate_viz(variances, normalized, max_dist, memory_depth):
    max_var = max(variances)

    bars_html = ""
    for k in range(1, max_dist + 1):
        height = 100 * normalized[k-1]
        color = "#3fb950" if k <= memory_depth else "#8b949e"
        bars_html += f'''
            <div class="bar-container">
                <div class="bar" style="height: {height:.1f}%; background: {color};"></div>
                <div class="label">{k}</div>
            </div>'''

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>RNN Memory Depth - Hutter Project</title>
    <style>
        body {{ font-family: system-ui, sans-serif; background: #0d1117; color: #c9d1d9; padding: 40px; margin: 0; }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        h1 {{ color: #58a6ff; }}
        h2 {{ color: #8b949e; font-size: 1em; margin-top: 30px; }}
        .back {{ display: inline-block; margin-bottom: 20px; padding: 8px 12px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; color: #58a6ff; text-decoration: none; }}
        .back:hover {{ background: #21262d; }}
        .chart {{ display: flex; align-items: flex-end; height: 200px; gap: 4px; margin: 20px 0; padding: 20px; background: #161b22; border-radius: 8px; }}
        .bar-container {{ display: flex; flex-direction: column; align-items: center; flex: 1; height: 100%; }}
        .bar {{ width: 100%; min-height: 2px; border-radius: 2px 2px 0 0; }}
        .label {{ font-size: 10px; color: #8b949e; margin-top: 4px; }}
        .result {{ background: #0d2818; border: 1px solid #238636; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .prediction {{ background: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        table {{ border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #30363d; padding: 8px 12px; text-align: left; }}
        th {{ background: #161b22; }}
        .method {{ background: #161b22; padding: 15px; border-radius: 8px; margin: 20px 0; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <a href="./" class="back">← Back to Archive</a>

        <h1>RNN Memory Depth</h1>
        <p>How far back does the RNN use context? Measured by variance of predictions across different values at distance k.</p>

        <div class="method">
            <strong>Method:</strong> Group samples by byte at position -k. Compute mean prediction for each group.
            Measure variance of means across groups. Higher variance = prediction depends more on that position.
        </div>

        <div class="result">
            <strong>Measured memory depth: ~{memory_depth} characters</strong><br>
            (Distance where dependency drops to 1/e of maximum)
        </div>

        <div class="prediction">
            <strong>Predicted:</strong> d_max = 24 / H_avg ≈ 24 / 2 = <strong>12 characters</strong>
        </div>

        <h2>Dependency vs Distance</h2>
        <p>Y-axis: Variance of prediction across different values at distance k. Higher = more dependency.</p>

        <div class="chart">
            {bars_html}
        </div>
        <p style="text-align: center; color: #8b949e;">Distance k (characters back)</p>

        <h2>Data</h2>
        <table>
            <tr><th>Distance</th><th>Variance</th><th>Normalized</th></tr>'''

    for k in range(1, min(15, max_dist + 1)):
        html += f'''
            <tr><td>{k}</td><td>{variances[k-1]:.6f}</td><td>{normalized[k-1]:.3f}</td></tr>'''

    html += f'''
        </table>

        <a href="./" class="back" style="margin-top: 30px;">← Back to Archive</a>
    </div>
</body>
</html>'''

    with open('/var/www/cmpr.ai/hutter/archive/20260131_4/memory-depth.html', 'w') as f:
        f.write(html)
    print("\n  Visualization saved to memory-depth.html")

if __name__ == '__main__':
    main()
