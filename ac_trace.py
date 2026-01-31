#!/usr/bin/env python3
"""
Arithmetic Coding Trace - Side by side with RNN

Shows how arithmetic coding maintains its interval [low, high)
in parallel with how the RNN maintains its hidden state h.

Key correspondence:
  AC interval [low, high) ↔ RNN hidden state h
  Interval width (high-low) ↔ remaining precision capacity
  -log2(width) ↔ accumulated bits ↔ ||h|| ?
"""

import numpy as np
import json

INPUT_SIZE = 256
HIDDEN_SIZE = 128
OUTPUT_SIZE = 256

# Arithmetic coding parameters
PRECISION_BITS = 32
FULL = 2 ** PRECISION_BITS
HALF = FULL // 2
QUARTER = FULL // 4

def load_model(path='model.bin'):
    with open(path, 'rb') as f:
        W_ih = np.frombuffer(f.read(HIDDEN_SIZE * INPUT_SIZE * 4), dtype=np.float32).reshape(HIDDEN_SIZE, INPUT_SIZE)
        b_h = np.frombuffer(f.read(HIDDEN_SIZE * 4), dtype=np.float32)
        W_hh = np.frombuffer(f.read(HIDDEN_SIZE * HIDDEN_SIZE * 4), dtype=np.float32).reshape(HIDDEN_SIZE, HIDDEN_SIZE)
        W_ho = np.frombuffer(f.read(OUTPUT_SIZE * HIDDEN_SIZE * 4), dtype=np.float32).reshape(OUTPUT_SIZE, HIDDEN_SIZE)
        b_o = np.frombuffer(f.read(OUTPUT_SIZE * 4), dtype=np.float32)
    return {'W_ih': W_ih, 'W_hh': W_hh, 'b_h': b_h, 'W_ho': W_ho, 'b_o': b_o}

def forward_step(model, h, byte_val):
    """Single RNN step."""
    x = np.zeros(INPUT_SIZE, dtype=np.float32)
    x[byte_val] = 1.0
    h_new = np.tanh(model['W_ih'] @ x + model['W_hh'] @ h + model['b_h'])
    logits = model['W_ho'] @ h_new + model['b_o']
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return h_new, probs

class ArithmeticCoder:
    """Simple arithmetic coder for comparison."""

    def __init__(self):
        self.low = 0
        self.high = FULL - 1
        self.bits_output = 0
        self.pending_bits = 0
        self.total_log_prob = 0.0  # Track cumulative -log2(p)

    def encode_symbol(self, probs, symbol):
        """Encode one symbol given probability distribution."""
        # Track cumulative information
        self.total_log_prob += -np.log2(probs[symbol] + 1e-10)

        # Build CDF
        cdf = np.zeros(len(probs) + 1)
        cdf[1:] = np.cumsum(probs)

        # Scale to integer range
        range_size = self.high - self.low + 1
        sym_low = int(cdf[symbol] * range_size)
        sym_high = int(cdf[symbol + 1] * range_size)

        # Update interval
        self.high = self.low + sym_high - 1
        self.low = self.low + sym_low

        # Output bits and renormalize
        while True:
            if self.high < HALF:
                self.output_bit(0)
                self.low = 2 * self.low
                self.high = 2 * self.high + 1
            elif self.low >= HALF:
                self.output_bit(1)
                self.low = 2 * (self.low - HALF)
                self.high = 2 * (self.high - HALF) + 1
            elif self.low >= QUARTER and self.high < 3 * QUARTER:
                self.pending_bits += 1
                self.low = 2 * (self.low - QUARTER)
                self.high = 2 * (self.high - QUARTER) + 1
            else:
                break

    def output_bit(self, bit):
        self.bits_output += 1 + self.pending_bits
        self.pending_bits = 0

    def get_interval_width(self):
        return (self.high - self.low + 1) / FULL

    def get_total_bits(self):
        """Total bits accumulated = bits output + pending + current interval bits."""
        return self.total_log_prob

    def get_interval_bits(self):
        """Just the current interval contribution."""
        width = self.get_interval_width()
        if width > 0:
            return -np.log2(width)
        return PRECISION_BITS

def collect_dual_trace(model, text_bytes, max_len=100):
    """
    Process text through both RNN and arithmetic coder in parallel.
    Track their states side by side.
    """
    h = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    ac = ArithmeticCoder()

    trace = {
        'chars': [],
        # RNN state
        'h_norms': [],
        'h_components': [],  # First few components
        # AC state
        'ac_low': [],
        'ac_high': [],
        'ac_width': [],
        'ac_bits': [],
        # Predictions (shared)
        'probs': [],
        'entropies': [],
        'surprisals': []
    }

    text = text_bytes[:max_len]

    for byte_val in text:
        # Get RNN prediction
        logits = model['W_ho'] @ h + model['b_o']
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        # Entropy and surprisal
        eps = 1e-10
        entropy = -np.sum(probs * np.log2(probs + eps))
        surprisal = -np.log2(probs[byte_val] + eps)

        # Record AC state before update
        trace['ac_low'].append(ac.low / FULL)
        trace['ac_high'].append(ac.high / FULL)
        trace['ac_width'].append(ac.get_interval_width())

        # Encode symbol in AC (this updates total_log_prob)
        ac.encode_symbol(probs, byte_val)

        # Record cumulative bits AFTER encoding this symbol
        trace['ac_bits'].append(ac.get_total_bits())

        # Record RNN state before update
        trace['h_norms'].append(float(np.linalg.norm(h)))
        trace['h_components'].append([float(x) for x in h[:4]])

        # Update RNN state
        h, _ = forward_step(model, h, byte_val)

        # Record common info
        try:
            char = chr(byte_val) if 32 <= byte_val < 127 else f'[{byte_val}]'
        except:
            char = f'[{byte_val}]'
        trace['chars'].append(char)
        trace['probs'].append([float(x) for x in probs[:10]])  # Top 10 probs
        trace['entropies'].append(float(entropy))
        trace['surprisals'].append(float(surprisal))

    return trace

def generate_dual_viz(trace, output_path):
    """Generate HTML showing RNN and AC states side by side."""

    n = len(trace['chars'])

    # JSON data
    chars_json = json.dumps(trace['chars'])
    h_norms_json = json.dumps(trace['h_norms'])
    ac_low_json = json.dumps(trace['ac_low'])
    ac_high_json = json.dumps(trace['ac_high'])
    ac_width_json = json.dumps(trace['ac_width'])
    ac_bits_json = json.dumps([float(x) for x in trace['ac_bits']])
    entropies_json = json.dumps(trace['entropies'])
    surprisals_json = json.dumps(trace['surprisals'])

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>AC ↔ RNN Memory Trace</title>
    <style>
        body {{ font-family: system-ui, sans-serif; background: #0d1117; color: #c9d1d9; padding: 40px; margin: 0; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        h1 {{ color: #58a6ff; }}
        h2 {{ color: #8b949e; font-size: 1em; margin-top: 30px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
        .back {{ display: inline-block; margin-bottom: 20px; padding: 8px 12px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; color: #58a6ff; text-decoration: none; }}
        .back:hover {{ background: #21262d; }}
        .analogy {{ background: #161b22; padding: 20px; border-radius: 8px; margin: 20px 0; display: grid; grid-template-columns: 1fr 60px 1fr; gap: 10px; align-items: start; }}
        .analogy h3 {{ margin: 0 0 10px 0; color: #58a6ff; font-size: 0.95em; }}
        .analogy .arrow {{ text-align: center; font-size: 24px; color: #8b949e; padding-top: 30px; }}
        .analogy pre {{ background: #21262d; padding: 10px; border-radius: 4px; margin: 5px 0; font-size: 0.85em; overflow-x: auto; }}
        .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .chart-container {{ background: #161b22; border-radius: 8px; padding: 20px; }}
        .chart-container h3 {{ color: #8b949e; font-size: 0.9em; margin: 0 0 15px 0; }}
        canvas {{ display: block; width: 100%; }}
        .insight {{ background: #0d2818; border: 1px solid #238636; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .text-display {{ font-family: monospace; background: #161b22; padding: 15px; border-radius: 8px; margin: 20px 0; line-height: 2; font-size: 0.95em; }}
        .step {{ display: inline-block; cursor: pointer; padding: 2px 4px; margin: 1px; border-radius: 3px; }}
        .step:hover {{ background: #30363d; }}
        .step.selected {{ background: #1f6feb; }}
        #detail {{ background: #161b22; padding: 15px; border-radius: 8px; margin: 20px 0; display: none; }}
        #detail table {{ width: 100%; border-collapse: collapse; }}
        #detail td, #detail th {{ border: 1px solid #30363d; padding: 8px; text-align: left; }}
        #detail th {{ background: #21262d; }}
        .metric {{ display: inline-block; margin: 0 10px; }}
        .metric .value {{ font-size: 1.5em; color: #58a6ff; }}
        .metric .label {{ font-size: 0.8em; color: #8b949e; }}
    </style>
</head>
<body>
    <div class="container">
        <a href="./" class="back">← Back to Archive</a>

        <h1>Arithmetic Coding ↔ RNN Memory</h1>
        <p>Side-by-side comparison of how AC and RNN carry context through time.</p>

        <div class="analogy">
            <div>
                <h3>Arithmetic Coding</h3>
                <pre>State: [low, high) ⊂ [0, 1)</pre>
                <pre>Update: narrow interval by p(symbol)</pre>
                <pre>Width shrinks: width *= p</pre>
                <pre>Bits used: -log₂(width)</pre>
                <pre>Precision limit: 32-64 bits</pre>
            </div>
            <div class="arrow">↔</div>
            <div>
                <h3>RNN Hidden State</h3>
                <pre>State: h ∈ ℝ¹²⁸ (within [-1,1] via tanh)</pre>
                <pre>Update: h' = tanh(W·x + U·h + b)</pre>
                <pre>Capacity shrinks: ||h|| changes</pre>
                <pre>Bits used: -log₂ p(output)</pre>
                <pre>Precision limit: float32 = 24 mantissa bits</pre>
            </div>
        </div>

        <div style="text-align: center; margin: 20px 0;">
            <div class="metric"><div class="value">{n}</div><div class="label">Steps</div></div>
            <div class="metric"><div class="value">{np.mean(trace['entropies']):.1f}</div><div class="label">Avg entropy</div></div>
            <div class="metric"><div class="value">{np.mean(trace['surprisals']):.1f}</div><div class="label">Avg surprisal</div></div>
            <div class="metric"><div class="value">{trace['ac_bits'][-1]:.1f}</div><div class="label">Total AC bits</div></div>
        </div>

        <h2>Click a character to see state details</h2>
        <div class="text-display" id="text-display"></div>

        <div id="detail">
            <h3 id="detail-title">Step N: 'x'</h3>
            <table>
                <tr><th></th><th>Arithmetic Coding</th><th>RNN</th></tr>
                <tr>
                    <td>State</td>
                    <td id="ac-state">[low, high)</td>
                    <td id="rnn-state">||h|| = ...</td>
                </tr>
                <tr>
                    <td>Width / Capacity</td>
                    <td id="ac-width">...</td>
                    <td id="rnn-capacity">...</td>
                </tr>
                <tr>
                    <td>Bits accumulated</td>
                    <td id="ac-bits">...</td>
                    <td id="rnn-bits">...</td>
                </tr>
                <tr>
                    <td>Entropy (uncertainty)</td>
                    <td colspan="2" id="entropy">...</td>
                </tr>
                <tr>
                    <td>Surprisal (this symbol)</td>
                    <td colspan="2" id="surprisal">...</td>
                </tr>
            </table>
        </div>

        <div class="chart-row">
            <div class="chart-container">
                <h3>AC Interval [low, high) over time</h3>
                <canvas id="ac-interval-chart" height="150"></canvas>
            </div>
            <div class="chart-container">
                <h3>RNN Hidden State ||h|| over time</h3>
                <canvas id="rnn-norm-chart" height="150"></canvas>
            </div>
        </div>

        <div class="chart-row">
            <div class="chart-container">
                <h3>AC: -log₂(interval width) = cumulative bits</h3>
                <canvas id="ac-bits-chart" height="150"></canvas>
            </div>
            <div class="chart-container">
                <h3>Entropy & Surprisal (shared prediction)</h3>
                <canvas id="entropy-chart" height="150"></canvas>
            </div>
        </div>

        <div class="insight">
            <strong>Key Insight:</strong> Both systems narrow down possibilities over time.
            AC does it explicitly (interval shrinks). RNN does it implicitly (hidden state encodes context).
            Both hit precision limits: AC after ~32-64 bits, RNN after ~24 bits × 128 dims ≈ 3000 bits (but not all usable due to correlation).
        </div>

        <a href="./" class="back" style="margin-top: 30px;">← Back to Archive</a>
    </div>

    <script>
        const chars = {chars_json};
        const hNorms = {h_norms_json};
        const acLow = {ac_low_json};
        const acHigh = {ac_high_json};
        const acWidth = {ac_width_json};
        const acBits = {ac_bits_json};
        const entropies = {entropies_json};
        const surprisals = {surprisals_json};
        const n = chars.length;

        // Populate text display
        const textDisplay = document.getElementById('text-display');
        let html = '';
        for (let i = 0; i < n; i++) {{
            let char = chars[i];
            if (char === '<') char = '&lt;';
            if (char === '>') char = '&gt;';
            if (char === '&') char = '&amp;';
            if (char === ' ') char = '&nbsp;';
            if (char === '\\n' || char === '[10]') char = '↵';
            html += `<span class="step" data-i="${{i}}">${{char}}</span>`;
        }}
        textDisplay.innerHTML = html;

        // Click handler for details
        let selectedStep = -1;
        textDisplay.addEventListener('click', (e) => {{
            if (e.target.classList.contains('step')) {{
                const i = parseInt(e.target.dataset.i);
                showDetail(i);

                // Update selection
                document.querySelectorAll('.step').forEach(s => s.classList.remove('selected'));
                e.target.classList.add('selected');
            }}
        }});

        function showDetail(i) {{
            const detail = document.getElementById('detail');
            detail.style.display = 'block';

            document.getElementById('detail-title').textContent = `Step ${{i+1}}: '${{chars[i]}}'`;
            document.getElementById('ac-state').textContent = `[${{acLow[i].toFixed(6)}}, ${{acHigh[i].toFixed(6)}})`;
            document.getElementById('rnn-state').textContent = `||h|| = ${{hNorms[i].toFixed(3)}}`;
            document.getElementById('ac-width').textContent = `width = ${{acWidth[i].toExponential(3)}}`;
            document.getElementById('rnn-capacity').textContent = `128 dims, tanh bounded`;
            document.getElementById('ac-bits').textContent = `${{acBits[i].toFixed(2)}} bits`;
            document.getElementById('rnn-bits').textContent = `~${{(24 * 128).toFixed(0)}} bits (theoretical max)`;
            document.getElementById('entropy').textContent = `${{entropies[i].toFixed(2)}} bits`;
            document.getElementById('surprisal').textContent = `${{surprisals[i].toFixed(2)}} bits`;
        }}

        // Chart helper
        function drawLineChart(canvasId, data, color, yMin, yMax) {{
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            const W = canvas.width, H = canvas.height;
            const pad = 30;

            // Axes
            ctx.strokeStyle = '#30363d';
            ctx.beginPath();
            ctx.moveTo(pad, H - pad);
            ctx.lineTo(W - pad, H - pad);
            ctx.stroke();

            // Line
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            const range = yMax - yMin || 1;
            for (let i = 0; i < data.length; i++) {{
                const x = pad + i * (W - 2*pad) / (data.length - 1);
                const y = H - pad - (data[i] - yMin) / range * (H - 2*pad);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }}
            ctx.stroke();
        }}

        function drawIntervalChart(canvasId) {{
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            const W = canvas.width, H = canvas.height;
            const pad = 30;

            // Axes
            ctx.strokeStyle = '#30363d';
            ctx.beginPath();
            ctx.moveTo(pad, H - pad);
            ctx.lineTo(W - pad, H - pad);
            ctx.stroke();

            // Draw interval as a band
            ctx.fillStyle = 'rgba(88, 166, 255, 0.3)';
            ctx.beginPath();
            for (let i = 0; i < n; i++) {{
                const x = pad + i * (W - 2*pad) / (n - 1);
                const yLow = H - pad - acLow[i] * (H - 2*pad);
                if (i === 0) ctx.moveTo(x, yLow);
                else ctx.lineTo(x, yLow);
            }}
            for (let i = n - 1; i >= 0; i--) {{
                const x = pad + i * (W - 2*pad) / (n - 1);
                const yHigh = H - pad - acHigh[i] * (H - 2*pad);
                ctx.lineTo(x, yHigh);
            }}
            ctx.closePath();
            ctx.fill();

            // Draw low and high lines
            ctx.strokeStyle = '#3fb950';
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = 0; i < n; i++) {{
                const x = pad + i * (W - 2*pad) / (n - 1);
                const y = H - pad - acLow[i] * (H - 2*pad);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }}
            ctx.stroke();

            ctx.strokeStyle = '#f85149';
            ctx.beginPath();
            for (let i = 0; i < n; i++) {{
                const x = pad + i * (W - 2*pad) / (n - 1);
                const y = H - pad - acHigh[i] * (H - 2*pad);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }}
            ctx.stroke();

            // Y labels
            ctx.fillStyle = '#8b949e';
            ctx.font = '10px system-ui';
            ctx.fillText('0', 5, H - pad);
            ctx.fillText('1', 5, pad);
        }}

        // Draw charts
        drawIntervalChart('ac-interval-chart');
        drawLineChart('rnn-norm-chart', hNorms, '#58a6ff', 0, Math.max(...hNorms) * 1.1);
        drawLineChart('ac-bits-chart', acBits, '#ffa36b', 0, Math.max(...acBits) * 1.1);

        // Entropy chart with both lines
        (function() {{
            const canvas = document.getElementById('entropy-chart');
            const ctx = canvas.getContext('2d');
            const W = canvas.width, H = canvas.height;
            const pad = 30;

            ctx.strokeStyle = '#30363d';
            ctx.beginPath();
            ctx.moveTo(pad, H - pad);
            ctx.lineTo(W - pad, H - pad);
            ctx.stroke();

            const maxY = Math.max(...entropies, ...surprisals) * 1.1;

            // Entropy
            ctx.strokeStyle = '#58a6ff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = 0; i < n; i++) {{
                const x = pad + i * (W - 2*pad) / (n - 1);
                const y = H - pad - entropies[i] / maxY * (H - 2*pad);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }}
            ctx.stroke();

            // Surprisal
            ctx.strokeStyle = '#f85149';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < n; i++) {{
                const x = pad + i * (W - 2*pad) / (n - 1);
                const y = H - pad - surprisals[i] / maxY * (H - 2*pad);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }}
            ctx.stroke();
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
        f.seek(10000)
        text_bytes = f.read(500)

    # Find clean start
    start = text_bytes.find(b'\n') + 1
    text_bytes = text_bytes[start:]

    print(f"Processing {min(100, len(text_bytes))} bytes...")
    trace = collect_dual_trace(model, text_bytes, max_len=100)

    output_path = '/var/www/cmpr.ai/hutter/archive/20260131_6/ac-rnn-trace.html'

    import os
    os.makedirs('/var/www/cmpr.ai/hutter/archive/20260131_6', exist_ok=True)

    generate_dual_viz(trace, output_path)

    print("\n=== DUAL TRACE SUMMARY ===")
    print(f"  Steps: {len(trace['chars'])}")
    print(f"  AC interval width at end: {trace['ac_width'][-1]:.2e}")
    print(f"  AC bits used: {trace['ac_bits'][-1]:.1f}")
    print(f"  Total surprisal: {sum(trace['surprisals']):.1f} bits")
    print(f"  Avg RNN ||h||: {np.mean(trace['h_norms']):.3f}")

if __name__ == '__main__':
    main()
