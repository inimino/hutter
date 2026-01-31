#!/usr/bin/env python3
"""
Memory Trace Visualization

Tracks the RNN hidden state h_t as we process text, analogous to
how arithmetic coding tracks its interval [low, high).

Key analogy:
  AC interval [low, high) ↔ RNN hidden state h
  Interval width ↔ "remaining capacity" / entropy
  Interval position ↔ accumulated context

This makes explicit how the RNN "carries entropy through time".
"""

import numpy as np
import json

def simple_pca(X, n_components=2):
    """Simple PCA implementation using numpy."""
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute covariance matrix
    cov = np.cov(X_centered, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Project onto top components
    components = eigenvectors[:, :n_components].T
    X_pca = X_centered @ eigenvectors[:, :n_components]

    # Variance explained
    total_var = np.sum(eigenvalues)
    var_explained = eigenvalues[:n_components] / total_var

    return X_pca, var_explained, components

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

def forward_step(model, h, byte_val):
    """Single RNN step. Returns new hidden state and output probabilities."""
    x = np.zeros(INPUT_SIZE, dtype=np.float32)
    x[byte_val] = 1.0
    h_new = np.tanh(model['W_ih'] @ x + model['W_hh'] @ h + model['b_h'])
    logits = model['W_ho'] @ h_new + model['b_o']
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return h_new, probs

def entropy(probs):
    """Entropy in bits."""
    eps = 1e-10
    return -np.sum(probs * np.log2(probs + eps))

def collect_memory_trace(model, text_bytes, max_len=200):
    """
    Process text and record the hidden state at each step.

    Returns:
        trace: dict with:
            - h_states: list of hidden state vectors
            - chars: list of input characters
            - entropies: list of output entropies
            - probs: list of probability distributions
            - surprisals: list of -log2(p(actual))
    """
    h = np.zeros(HIDDEN_SIZE, dtype=np.float32)

    trace = {
        'h_states': [h.copy()],  # Initial state
        'chars': [''],  # No input yet
        'entropies': [],
        'probs': [],
        'surprisals': []
    }

    text = text_bytes[:max_len]

    for i, byte_val in enumerate(text):
        # Get prediction before seeing this byte
        logits = model['W_ho'] @ h + model['b_o']
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        # Record entropy and surprisal
        trace['entropies'].append(entropy(probs))
        trace['probs'].append(probs.tolist())
        trace['surprisals'].append(-np.log2(probs[byte_val] + 1e-10))

        # Update hidden state
        h, _ = forward_step(model, h, byte_val)

        # Record
        trace['h_states'].append(h.copy())
        try:
            char = chr(byte_val) if 32 <= byte_val < 127 else f'[{byte_val}]'
        except:
            char = f'[{byte_val}]'
        trace['chars'].append(char)

    return trace

def analyze_trace(trace):
    """Compute summary statistics from trace."""
    h_states = np.array(trace['h_states'])

    # Hidden state statistics
    h_norms = np.linalg.norm(h_states, axis=1)
    h_means = np.mean(h_states, axis=1)
    h_stds = np.std(h_states, axis=1)

    # PCA of hidden states
    h_pca, var_explained, components = simple_pca(h_states, n_components=2)

    return {
        'h_norms': h_norms.tolist(),
        'h_means': h_means.tolist(),
        'h_stds': h_stds.tolist(),
        'h_pca': h_pca.tolist(),
        'pca_variance': var_explained.tolist(),
        'pca_components': components.tolist()
    }

def generate_viz(trace, analysis, text_sample, output_path):
    """Generate HTML visualization of memory trace."""

    n = len(trace['entropies'])

    # Prepare data for charts
    pca_points = analysis['h_pca']
    entropies = trace['entropies']
    surprisals = trace['surprisals']
    chars = trace['chars'][1:]  # Skip initial empty

    # Create PCA path data
    pca_json = json.dumps(pca_points)

    # Create entropy/surprisal bar data (convert to native Python floats)
    entropy_json = json.dumps([float(x) for x in entropies])
    surprisal_json = json.dumps([float(x) for x in surprisals])
    chars_json = json.dumps(chars)

    # Sample of hidden state components over time (first 8 dims)
    h_states = np.array(trace['h_states'])
    h_sample = h_states[:, :8].tolist()
    h_sample_json = json.dumps(h_sample)

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Memory Trace - Hutter RNN</title>
    <style>
        body {{ font-family: system-ui, sans-serif; background: #0d1117; color: #c9d1d9; padding: 40px; margin: 0; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #58a6ff; }}
        h2 {{ color: #8b949e; font-size: 1em; margin-top: 30px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
        .back {{ display: inline-block; margin-bottom: 20px; padding: 8px 12px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; color: #58a6ff; text-decoration: none; }}
        .back:hover {{ background: #21262d; }}
        .info {{ background: #161b22; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .chart-container {{ background: #161b22; border-radius: 8px; padding: 20px; margin: 20px 0; }}
        canvas {{ display: block; margin: 0 auto; }}
        .text-display {{ font-family: monospace; background: #161b22; padding: 15px; border-radius: 8px; overflow-x: auto; white-space: pre; margin: 20px 0; line-height: 1.8; }}
        .char {{ display: inline-block; padding: 2px 4px; margin: 1px; border-radius: 3px; cursor: pointer; }}
        .char:hover {{ background: #30363d; }}
        .legend {{ display: flex; gap: 20px; margin: 10px 0; font-size: 0.9em; }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .legend-color {{ width: 12px; height: 12px; border-radius: 2px; }}
        #tooltip {{ position: fixed; background: #21262d; border: 1px solid #30363d; padding: 8px 12px; border-radius: 6px; font-size: 0.85em; pointer-events: none; display: none; z-index: 1000; }}
        .insight {{ background: #0d2818; border: 1px solid #238636; padding: 15px; border-radius: 8px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <a href="./" class="back">← Back to Archive</a>

        <h1>Memory Trace Visualization</h1>
        <p>Tracking RNN hidden state h through text, analogous to arithmetic coding interval.</p>

        <div class="info">
            <strong>Analogy:</strong> AC interval [low, high) ↔ RNN hidden state h<br>
            <strong>Sample:</strong> {n} characters | <strong>Hidden dim:</strong> 128 | <strong>PCA variance:</strong> {analysis['pca_variance'][0]*100:.1f}% + {analysis['pca_variance'][1]*100:.1f}%
        </div>

        <h2>Hidden State Trajectory (PCA)</h2>
        <p>The RNN's 128D hidden state projected to 2D. Each point is one timestep. Color: blue→red over time.</p>
        <div class="chart-container">
            <canvas id="pca-chart" width="600" height="400"></canvas>
        </div>

        <h2>Entropy Over Time</h2>
        <p>Output entropy (uncertainty) at each step. Lower = more confident prediction.</p>
        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background: #58a6ff;"></div> Entropy (bits)</div>
            <div class="legend-item"><div class="legend-color" style="background: #f85149;"></div> Surprisal (-log₂ p)</div>
        </div>
        <div class="chart-container">
            <canvas id="entropy-chart" width="800" height="200"></canvas>
        </div>

        <h2>Hidden State Components</h2>
        <p>First 8 dimensions of h over time. Shows how each component evolves.</p>
        <div class="chart-container">
            <canvas id="components-chart" width="800" height="250"></canvas>
        </div>

        <h2>Text with Entropy Coloring</h2>
        <p>Hover to see entropy. Red = high surprisal, Green = well predicted.</p>
        <div class="text-display" id="text-display"></div>

        <div class="insight">
            <strong>Key Insight:</strong> Like arithmetic coding's interval, the hidden state h accumulates context.
            High-entropy predictions "use up" precision. The trajectory shows how context flows through time.
        </div>

        <a href="./" class="back" style="margin-top: 30px;">← Back to Archive</a>
    </div>

    <div id="tooltip"></div>

    <script>
        const pcaData = {pca_json};
        const entropyData = {entropy_json};
        const surprisalData = {surprisal_json};
        const charsData = {chars_json};
        const hComponents = {h_sample_json};

        // PCA Chart
        (function() {{
            const canvas = document.getElementById('pca-chart');
            const ctx = canvas.getContext('2d');
            const W = canvas.width, H = canvas.height;
            const pad = 40;

            // Find bounds
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            for (const [x, y] of pcaData) {{
                minX = Math.min(minX, x); maxX = Math.max(maxX, x);
                minY = Math.min(minY, y); maxY = Math.max(maxY, y);
            }}
            const rangeX = maxX - minX || 1;
            const rangeY = maxY - minY || 1;

            function toScreen(x, y) {{
                return [
                    pad + (x - minX) / rangeX * (W - 2*pad),
                    H - pad - (y - minY) / rangeY * (H - 2*pad)
                ];
            }}

            // Draw axes
            ctx.strokeStyle = '#30363d';
            ctx.beginPath();
            ctx.moveTo(pad, H - pad); ctx.lineTo(W - pad, H - pad);
            ctx.moveTo(pad, H - pad); ctx.lineTo(pad, pad);
            ctx.stroke();

            // Draw path
            ctx.lineWidth = 1;
            ctx.beginPath();
            const [sx, sy] = toScreen(pcaData[0][0], pcaData[0][1]);
            ctx.moveTo(sx, sy);
            for (let i = 1; i < pcaData.length; i++) {{
                const [px, py] = toScreen(pcaData[i][0], pcaData[i][1]);
                ctx.strokeStyle = `rgba(88, 166, 255, 0.3)`;
                ctx.lineTo(px, py);
            }}
            ctx.stroke();

            // Draw points with color gradient
            for (let i = 0; i < pcaData.length; i++) {{
                const [px, py] = toScreen(pcaData[i][0], pcaData[i][1]);
                const t = i / (pcaData.length - 1);
                const r = Math.floor(88 + t * (248 - 88));
                const g = Math.floor(166 - t * (166 - 81));
                const b = Math.floor(255 - t * (255 - 73));
                ctx.fillStyle = `rgb(${{r}}, ${{g}}, ${{b}})`;
                ctx.beginPath();
                ctx.arc(px, py, 3, 0, Math.PI * 2);
                ctx.fill();
            }}

            // Labels
            ctx.fillStyle = '#8b949e';
            ctx.font = '11px system-ui';
            ctx.fillText('PC1', W/2, H - 10);
            ctx.save();
            ctx.translate(15, H/2);
            ctx.rotate(-Math.PI/2);
            ctx.fillText('PC2', 0, 0);
            ctx.restore();

            // Start/end markers
            ctx.fillStyle = '#3fb950';
            const [startX, startY] = toScreen(pcaData[0][0], pcaData[0][1]);
            ctx.beginPath(); ctx.arc(startX, startY, 6, 0, Math.PI*2); ctx.fill();
            ctx.fillStyle = '#f85149';
            const [endX, endY] = toScreen(pcaData[pcaData.length-1][0], pcaData[pcaData.length-1][1]);
            ctx.beginPath(); ctx.arc(endX, endY, 6, 0, Math.PI*2); ctx.fill();
        }})();

        // Entropy Chart
        (function() {{
            const canvas = document.getElementById('entropy-chart');
            const ctx = canvas.getContext('2d');
            const W = canvas.width, H = canvas.height;
            const pad = 40;
            const n = entropyData.length;

            const maxE = Math.max(...entropyData);
            const maxS = Math.max(...surprisalData);
            const maxY = Math.max(maxE, maxS);

            // Draw axes
            ctx.strokeStyle = '#30363d';
            ctx.beginPath();
            ctx.moveTo(pad, H - pad); ctx.lineTo(W - pad, H - pad);
            ctx.moveTo(pad, H - pad); ctx.lineTo(pad, pad);
            ctx.stroke();

            const barW = (W - 2*pad) / n * 0.4;

            // Draw entropy bars
            ctx.fillStyle = 'rgba(88, 166, 255, 0.7)';
            for (let i = 0; i < n; i++) {{
                const x = pad + i * (W - 2*pad) / n;
                const h = entropyData[i] / maxY * (H - 2*pad);
                ctx.fillRect(x, H - pad - h, barW, h);
            }}

            // Draw surprisal line
            ctx.strokeStyle = '#f85149';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < n; i++) {{
                const x = pad + i * (W - 2*pad) / n + barW;
                const y = H - pad - surprisalData[i] / maxY * (H - 2*pad);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }}
            ctx.stroke();

            // Y-axis labels
            ctx.fillStyle = '#8b949e';
            ctx.font = '10px system-ui';
            ctx.fillText('0', pad - 15, H - pad);
            ctx.fillText(maxY.toFixed(1), pad - 25, pad + 10);
        }})();

        // Components Chart
        (function() {{
            const canvas = document.getElementById('components-chart');
            const ctx = canvas.getContext('2d');
            const W = canvas.width, H = canvas.height;
            const pad = 40;
            const n = hComponents.length;
            const nDims = 8;

            // Colors for each dimension
            const colors = ['#ff6b6b', '#ffa36b', '#ffd56b', '#98d98e', '#6bc5ff', '#a56bff', '#ff6bda', '#8b949e'];

            // Find global min/max
            let minV = Infinity, maxV = -Infinity;
            for (const h of hComponents) {{
                for (let d = 0; d < nDims; d++) {{
                    minV = Math.min(minV, h[d]);
                    maxV = Math.max(maxV, h[d]);
                }}
            }}
            const rangeV = maxV - minV || 1;

            // Draw axes
            ctx.strokeStyle = '#30363d';
            ctx.beginPath();
            ctx.moveTo(pad, H - pad); ctx.lineTo(W - pad, H - pad);
            ctx.moveTo(pad, H - pad); ctx.lineTo(pad, pad);
            ctx.stroke();

            // Draw zero line
            const zeroY = H - pad - (0 - minV) / rangeV * (H - 2*pad);
            ctx.strokeStyle = '#30363d';
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(pad, zeroY);
            ctx.lineTo(W - pad, zeroY);
            ctx.stroke();
            ctx.setLineDash([]);

            // Draw each dimension
            for (let d = 0; d < nDims; d++) {{
                ctx.strokeStyle = colors[d];
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                for (let i = 0; i < n; i++) {{
                    const x = pad + i * (W - 2*pad) / (n - 1);
                    const y = H - pad - (hComponents[i][d] - minV) / rangeV * (H - 2*pad);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }}
                ctx.stroke();
            }}

            // Legend
            ctx.font = '10px system-ui';
            for (let d = 0; d < nDims; d++) {{
                ctx.fillStyle = colors[d];
                ctx.fillRect(W - 100 + (d % 4) * 25, d < 4 ? 10 : 25, 8, 8);
                ctx.fillStyle = '#8b949e';
                ctx.fillText('h' + d, W - 90 + (d % 4) * 25, d < 4 ? 18 : 33);
            }}
        }})();

        // Text display with coloring
        (function() {{
            const container = document.getElementById('text-display');
            const tooltip = document.getElementById('tooltip');

            let html = '';
            for (let i = 0; i < charsData.length; i++) {{
                const s = surprisalData[i];
                const e = entropyData[i];
                // Color: green (low surprisal) to red (high surprisal)
                const maxS = Math.max(...surprisalData);
                const t = Math.min(s / maxS, 1);
                const r = Math.floor(63 + t * (248 - 63));
                const g = Math.floor(185 - t * (185 - 81));
                const b = Math.floor(80 - t * (80 - 73));

                let char = charsData[i];
                if (char === '<') char = '&lt;';
                if (char === '>') char = '&gt;';
                if (char === '&') char = '&amp;';
                if (char === ' ') char = '&nbsp;';
                if (char === '\\n' || char === '[10]') char = '↵<br>';

                html += `<span class="char" style="background: rgba(${{r}}, ${{g}}, ${{b}}, 0.3);" data-i="${{i}}">` + char + '</span>';
            }}
            container.innerHTML = html;

            container.addEventListener('mouseover', (e) => {{
                if (e.target.classList.contains('char')) {{
                    const i = parseInt(e.target.dataset.i);
                    tooltip.innerHTML = `<strong>${{charsData[i]}}</strong><br>Entropy: ${{entropyData[i].toFixed(2)}} bits<br>Surprisal: ${{surprisalData[i].toFixed(2)}} bits`;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (e.clientX + 10) + 'px';
                    tooltip.style.top = (e.clientY + 10) + 'px';
                }}
            }});

            container.addEventListener('mouseout', () => {{
                tooltip.style.display = 'none';
            }});
        }})();
    </script>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"  Visualization saved to {output_path}")

def main():
    print("Loading model...")
    model = load_model('model.bin')

    print("Loading data...")
    with open('enwik9', 'rb') as f:
        # Skip to find a good passage
        f.seek(10000)
        text_bytes = f.read(500)

    # Find a clean starting point (after a newline)
    start = text_bytes.find(b'\n') + 1
    text_bytes = text_bytes[start:]

    print(f"Processing {len(text_bytes)} bytes...")
    trace = collect_memory_trace(model, text_bytes, max_len=150)

    print("Analyzing trace...")
    analysis = analyze_trace(trace)

    print(f"PCA variance explained: {analysis['pca_variance'][0]*100:.1f}% + {analysis['pca_variance'][1]*100:.1f}%")

    # Text sample for display
    text_sample = bytes(text_bytes[:150]).decode('utf-8', errors='replace')

    output_path = '/var/www/cmpr.ai/hutter/archive/20260131_6/memory-trace.html'

    # Create archive directory if needed
    import os
    os.makedirs('/var/www/cmpr.ai/hutter/archive/20260131_6', exist_ok=True)

    generate_viz(trace, analysis, text_sample, output_path)

    # Also save raw data
    np.savez('/var/www/cmpr.ai/hutter/archive/20260131_6/memory-trace-data.npz',
             h_states=np.array(trace['h_states']),
             entropies=np.array(trace['entropies']),
             surprisals=np.array(trace['surprisals']),
             pca=np.array(analysis['h_pca']),
             pca_components=np.array(analysis['pca_components']))
    print("  Raw data saved to memory-trace-data.npz")

    # Print summary
    print("\n=== MEMORY TRACE SUMMARY ===")
    print(f"  Steps processed: {len(trace['entropies'])}")
    print(f"  Avg entropy: {np.mean(trace['entropies']):.2f} bits")
    print(f"  Avg surprisal: {np.mean(trace['surprisals']):.2f} bits")
    print(f"  Hidden state norm: {np.mean(analysis['h_norms']):.3f} ± {np.std(analysis['h_norms']):.3f}")

if __name__ == '__main__':
    main()
