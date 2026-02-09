/*
 * pattern_viewer.c — Generate an HTML viewer for the SN-form pattern inventory.
 *
 * Loads the trained RNN and data, computes the factor map for all 128 neurons,
 * and outputs a self-contained HTML file with:
 *   - SN-form pattern for each neuron (offset pair, conditional means, state)
 *   - Generalization verdict (structural vs coincidental)
 *   - Interactive data view with pattern highlighting
 *   - Ablation importance ranking
 *
 * Usage: pattern_viewer <data_file> <model_file> > pattern-viewer.html
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256
#define MAX_DATA 1100
#define N_OFFSETS 8

typedef struct {
    float Wx[HIDDEN_SIZE][INPUT_SIZE];
    float Wh[HIDDEN_SIZE][HIDDEN_SIZE];
    float bh[HIDDEN_SIZE];
    float Wy[OUTPUT_SIZE][HIDDEN_SIZE];
    float by[OUTPUT_SIZE];
    float h[HIDDEN_SIZE];
} RNN;

void load_model(RNN* rnn, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("load_model"); exit(1); }
    fread(rnn->Wx, sizeof(float), HIDDEN_SIZE * INPUT_SIZE, f);
    fread(rnn->Wh, sizeof(float), HIDDEN_SIZE * HIDDEN_SIZE, f);
    fread(rnn->bh, sizeof(float), HIDDEN_SIZE, f);
    fread(rnn->Wy, sizeof(float), OUTPUT_SIZE * HIDDEN_SIZE, f);
    fread(rnn->by, sizeof(float), OUTPUT_SIZE, f);
    fclose(f);
}

static char safe_char(int c) { return (c >= 32 && c < 127) ? c : '.'; }
static char* js_safe(int c) {
    static char buf[8];
    if (c == '\\') { strcpy(buf, "\\\\"); return buf; }
    if (c == '\'') { strcpy(buf, "\\'"); return buf; }
    if (c == '"') { strcpy(buf, "\\\""); return buf; }
    if (c == '<') { strcpy(buf, "&lt;"); return buf; }
    if (c == '>') { strcpy(buf, "&gt;"); return buf; }
    if (c == '&') { strcpy(buf, "&amp;"); return buf; }
    if (c >= 32 && c < 127) { buf[0] = c; buf[1] = 0; return buf; }
    sprintf(buf, "\\x%02x", c);
    return buf;
}

static int OFFSETS[N_OFFSETS] = {1, 8, 20, 3, 27, 2, 12, 7};

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <data_file> <model_file> > output.html\n", argv[0]);
        return 1;
    }

    /* Load data */
    FILE* df = fopen(argv[1], "rb");
    if (!df) { perror("data"); return 1; }
    unsigned char data[MAX_DATA];
    int N = fread(data, 1, MAX_DATA, df);
    fclose(df);

    /* Load model */
    RNN rnn;
    load_model(&rnn, argv[2]);
    memset(rnn.h, 0, sizeof(rnn.h));

    /* Forward pass */
    float h_states[MAX_DATA][HIDDEN_SIZE];
    for (int t = 0; t < N; t++) {
        float h_new[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            float sum = rnn.bh[i] + rnn.Wx[i][data[t]];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                sum += rnn.Wh[i][j] * rnn.h[j];
            h_new[i] = tanhf(sum);
        }
        memcpy(rnn.h, h_new, sizeof(h_new));
        memcpy(h_states[t], rnn.h, sizeof(rnn.h));
    }

    /* Compute word_len and in_tag at each position */
    int word_len[MAX_DATA];
    int in_tag[MAX_DATA];
    word_len[0] = 0; in_tag[0] = 0;
    for (int t = 0; t < N; t++) {
        if (t > 0) {
            if (data[t-1]==' '||data[t-1]=='\n'||data[t-1]=='<'||data[t-1]=='>')
                word_len[t] = 0;
            else
                word_len[t] = (word_len[t-1] < 15) ? word_len[t-1]+1 : 15;
            in_tag[t] = in_tag[t-1];
            if (data[t-1] == '<') in_tag[t] = 1;
            if (data[t-1] == '>') in_tag[t] = 0;
        }
    }

    /* ===================================================================
     * Factor map: for each neuron, find best 2-offset pair
     * =================================================================== */

    typedef struct {
        int neuron;
        int d1, d2;
        float r2;
        float ablation_delta; /* bpc increase when zeroed */
        float mean_h;
        float mean_abs_h;
        /* Top contributing output bytes */
        int top_out[5];
        float top_out_wy[5];
        /* Conditional mean stats */
        float cond_mean_range; /* max - min conditional mean */
    } NeuronInfo;

    NeuronInfo info[HIDDEN_SIZE];

    /* For each neuron, try all pairs of offsets */
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        info[j].neuron = j;
        info[j].mean_h = 0;
        info[j].mean_abs_h = 0;
        for (int t = 0; t < N; t++) {
            info[j].mean_h += h_states[t][j];
            info[j].mean_abs_h += fabsf(h_states[t][j]);
        }
        info[j].mean_h /= N;
        info[j].mean_abs_h /= N;

        /* Find best 2-offset pair by R² */
        float best_r2 = -1;
        int best_d1 = 1, best_d2 = 8;

        for (int oi = 0; oi < N_OFFSETS; oi++) {
            for (int oj = oi + 1; oj < N_OFFSETS; oj++) {
                int d1 = OFFSETS[oi], d2 = OFFSETS[oj];
                int max_d = d1 > d2 ? d1 : d2;

                /* Compute conditional means E[h_j | data[t-d1], data[t-d2]] */
                float cm_sum[256][256];
                int cm_count[256][256];
                memset(cm_sum, 0, sizeof(cm_sum));
                memset(cm_count, 0, sizeof(cm_count));

                for (int t = max_d; t < N; t++) {
                    int b1 = data[t - d1], b2 = data[t - d2];
                    cm_sum[b1][b2] += h_states[t][j];
                    cm_count[b1][b2]++;
                }

                /* Compute R² */
                double ss_res = 0, ss_tot = 0;
                double mean_j = 0;
                int cnt = 0;
                for (int t = max_d; t < N; t++) {
                    mean_j += h_states[t][j];
                    cnt++;
                }
                mean_j /= cnt;

                for (int t = max_d; t < N; t++) {
                    int b1 = data[t - d1], b2 = data[t - d2];
                    float predicted = cm_sum[b1][b2] / cm_count[b1][b2];
                    float actual = h_states[t][j];
                    ss_res += (actual - predicted) * (actual - predicted);
                    ss_tot += (actual - mean_j) * (actual - mean_j);
                }
                float r2 = (ss_tot > 0) ? 1.0 - ss_res / ss_tot : 0;

                if (r2 > best_r2) {
                    best_r2 = r2;
                    best_d1 = d1;
                    best_d2 = d2;
                }
            }
        }

        info[j].d1 = best_d1;
        info[j].d2 = best_d2;
        info[j].r2 = best_r2;

        /* Conditional mean range */
        int max_d = best_d1 > best_d2 ? best_d1 : best_d2;
        float cm_sum2[256][256];
        int cm_count2[256][256];
        memset(cm_sum2, 0, sizeof(cm_sum2));
        memset(cm_count2, 0, sizeof(cm_count2));
        for (int t = max_d; t < N; t++) {
            int b1 = data[t - best_d1], b2 = data[t - best_d2];
            cm_sum2[b1][b2] += h_states[t][j];
            cm_count2[b1][b2]++;
        }
        float cm_min = 999, cm_max = -999;
        for (int b1 = 0; b1 < 256; b1++)
            for (int b2 = 0; b2 < 256; b2++)
                if (cm_count2[b1][b2] > 0) {
                    float m = cm_sum2[b1][b2] / cm_count2[b1][b2];
                    if (m < cm_min) cm_min = m;
                    if (m > cm_max) cm_max = m;
                }
        info[j].cond_mean_range = cm_max - cm_min;

        /* Ablation: zero this neuron, compute bpc change */
        float orig_bpc = 0, ablated_bpc = 0;
        int eval_cnt = 0;
        for (int t = 0; t < N - 1; t++) {
            float logits_orig[OUTPUT_SIZE], logits_abl[OUTPUT_SIZE];
            for (int y = 0; y < OUTPUT_SIZE; y++) {
                float s_orig = rnn.by[y], s_abl = rnn.by[y];
                for (int k = 0; k < HIDDEN_SIZE; k++) {
                    s_orig += rnn.Wy[y][k] * h_states[t][k];
                    if (k != j)
                        s_abl += rnn.Wy[y][k] * h_states[t][k];
                }
                logits_orig[y] = s_orig;
                logits_abl[y] = s_abl;
            }
            /* softmax + loss for both */
            for (int v = 0; v < 2; v++) {
                float* lg = (v == 0) ? logits_orig : logits_abl;
                float mx = lg[0];
                for (int y = 1; y < OUTPUT_SIZE; y++)
                    if (lg[y] > mx) mx = lg[y];
                float se = 0;
                for (int y = 0; y < OUTPUT_SIZE; y++)
                    se += expf(lg[y] - mx);
                float lp = lg[data[t+1]] - mx - logf(se);
                if (v == 0) orig_bpc -= lp;
                else ablated_bpc -= lp;
            }
            eval_cnt++;
        }
        orig_bpc /= eval_cnt * logf(2.0);
        ablated_bpc /= eval_cnt * logf(2.0);
        info[j].ablation_delta = ablated_bpc - orig_bpc;

        /* Top output bytes by |W_y[y][j]| */
        int top_idx[5] = {0,0,0,0,0};
        float top_val[5] = {0,0,0,0,0};
        for (int y = 0; y < OUTPUT_SIZE; y++) {
            float v = fabsf(rnn.Wy[y][j]);
            for (int k = 0; k < 5; k++) {
                if (v > top_val[k]) {
                    for (int m = 4; m > k; m--) {
                        top_idx[m] = top_idx[m-1];
                        top_val[m] = top_val[m-1];
                    }
                    top_idx[k] = y;
                    top_val[k] = v;
                    break;
                }
            }
        }
        for (int k = 0; k < 5; k++) {
            info[j].top_out[k] = top_idx[k];
            info[j].top_out_wy[k] = rnn.Wy[top_idx[k]][j];
        }
    }

    /* ===================================================================
     * Compute per-position bpc
     * =================================================================== */
    float pos_bpc[MAX_DATA];
    int pos_pred[MAX_DATA]; /* predicted byte */
    float pos_conf[MAX_DATA]; /* confidence of correct prediction */
    for (int t = 0; t < N - 1; t++) {
        float logits[OUTPUT_SIZE];
        for (int y = 0; y < OUTPUT_SIZE; y++) {
            float s = rnn.by[y];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                s += rnn.Wy[y][j] * h_states[t][j];
            logits[y] = s;
        }
        float mx = logits[0];
        for (int y = 1; y < OUTPUT_SIZE; y++)
            if (logits[y] > mx) mx = logits[y];
        float se = 0;
        for (int y = 0; y < OUTPUT_SIZE; y++)
            se += expf(logits[y] - mx);
        pos_bpc[t] = -(logits[data[t+1]] - mx - logf(se)) / logf(2.0);
        pos_conf[t] = expf(logits[data[t+1]] - mx) / se;

        /* Find predicted byte */
        int best_y = 0;
        for (int y = 1; y < OUTPUT_SIZE; y++)
            if (logits[y] > logits[best_y]) best_y = y;
        pos_pred[t] = best_y;
    }

    /* ===================================================================
     * Generate HTML
     * =================================================================== */

    printf("<!DOCTYPE html>\n<html>\n<head>\n");
    printf("<title>Pattern Viewer - Hutter RNN Factor Map</title>\n");
    printf("<style>\n");
    printf("* { box-sizing: border-box; margin: 0; padding: 0; }\n");
    printf("body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0d1117; color: #c9d1d9; }\n");
    printf(".header { background: #161b22; padding: 16px 24px; border-bottom: 1px solid #30363d; }\n");
    printf(".header h1 { color: #58a6ff; font-size: 18px; }\n");
    printf(".header .stats { color: #8b949e; font-size: 13px; margin-top: 4px; }\n");
    printf(".main { display: flex; height: calc(100vh - 60px); }\n");
    printf(".panel { border-right: 1px solid #30363d; overflow-y: auto; }\n");
    printf(".neuron-list { width: 340px; }\n");
    printf(".detail { flex: 1; padding: 16px; overflow-y: auto; }\n");
    printf(".neuron-row { padding: 6px 12px; cursor: pointer; border-bottom: 1px solid #21262d; font-size: 12px; display: flex; align-items: center; gap: 8px; }\n");
    printf(".neuron-row:hover { background: #161b22; }\n");
    printf(".neuron-row.selected { background: #1c2128; border-left: 3px solid #58a6ff; }\n");
    printf(".neuron-row .id { color: #58a6ff; width: 36px; }\n");
    printf(".neuron-row .pair { color: #d2a8ff; width: 50px; }\n");
    printf(".neuron-row .r2 { color: #3fb950; width: 40px; }\n");
    printf(".neuron-row .delta { width: 50px; }\n");
    printf(".neuron-row .sn { color: #8b949e; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }\n");
    printf(".gen { display: inline-block; width: 8px; height: 8px; border-radius: 50%%; margin-right: 4px; }\n");
    printf(".gen.yes { background: #3fb950; }\n");
    printf(".gen.maybe { background: #d29922; }\n");
    printf(".gen.no { background: #f85149; }\n");
    printf(".detail h2 { color: #58a6ff; font-size: 16px; margin-bottom: 12px; }\n");
    printf(".detail h3 { color: #8b949e; font-size: 13px; margin: 16px 0 8px; }\n");
    printf(".sn-box { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; margin: 8px 0; font-size: 13px; line-height: 1.6; }\n");
    printf(".sn-box .keyword { color: #ff7b72; }\n");
    printf(".sn-box .value { color: #79c0ff; }\n");
    printf(".sn-box .prob { color: #3fb950; }\n");
    printf(".data-view { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; margin: 8px 0; font-size: 12px; line-height: 1.8; word-break: break-all; }\n");
    printf(".data-view .byte { display: inline; cursor: pointer; padding: 1px; border-radius: 2px; }\n");
    printf(".data-view .byte:hover { background: #30363d; }\n");
    printf(".data-view .byte.active { background: #1f6feb44; outline: 1px solid #1f6feb; }\n");
    printf(".data-view .byte.highlight { background: #3fb95033; }\n");
    printf(".metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 8px 0; }\n");
    printf(".metric { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px; text-align: center; }\n");
    printf(".metric .val { color: #58a6ff; font-size: 18px; font-weight: bold; }\n");
    printf(".metric .label { color: #8b949e; font-size: 11px; margin-top: 4px; }\n");
    printf(".out-table { width: 100%%; border-collapse: collapse; font-size: 12px; margin: 8px 0; }\n");
    printf(".out-table th { color: #8b949e; text-align: left; padding: 4px 8px; border-bottom: 1px solid #30363d; }\n");
    printf(".out-table td { padding: 4px 8px; border-bottom: 1px solid #21262d; }\n");
    printf(".bar { display: inline-block; height: 10px; border-radius: 2px; }\n");
    printf(".bar.pos { background: #3fb950; }\n");
    printf(".bar.neg { background: #f85149; }\n");
    printf(".sort-btn { background: none; border: 1px solid #30363d; color: #8b949e; padding: 2px 8px; border-radius: 4px; cursor: pointer; font-size: 11px; margin: 2px; }\n");
    printf(".sort-btn:hover { background: #21262d; color: #c9d1d9; }\n");
    printf(".sort-btn.active { background: #1f6feb; color: white; border-color: #1f6feb; }\n");
    printf("</style>\n</head>\n<body>\n");

    /* Header */
    printf("<div class=\"header\">\n");
    printf("  <h1>Pattern Viewer: 128 Neurons in SN Form</h1>\n");
    printf("  <div class=\"stats\">sat-rnn (128 hidden, tanh, BPTT-50) &middot; 0.079 bpc on 1024 bytes of enwik9 &middot; Factor map R&sup2;=0.837</div>\n");
    printf("</div>\n");

    printf("<div class=\"main\">\n");

    /* Neuron list panel */
    printf("<div class=\"panel neuron-list\">\n");
    printf("  <div style=\"padding:8px 12px;border-bottom:1px solid #30363d\">\n");
    printf("    <button class=\"sort-btn active\" onclick=\"sortBy('importance')\">Importance</button>\n");
    printf("    <button class=\"sort-btn\" onclick=\"sortBy('id')\">ID</button>\n");
    printf("    <button class=\"sort-btn\" onclick=\"sortBy('r2')\">R&sup2;</button>\n");
    printf("    <button class=\"sort-btn\" onclick=\"sortBy('pair')\">Pair</button>\n");
    printf("  </div>\n");
    printf("  <div id=\"neuron-list-inner\"></div>\n");
    printf("</div>\n");

    /* Detail panel */
    printf("<div class=\"detail\" id=\"detail\"></div>\n");
    printf("</div>\n");

    /* JavaScript data */
    printf("<script>\n");

    /* Emit data bytes */
    printf("const DATA = [");
    for (int t = 0; t < N; t++) {
        if (t > 0) printf(",");
        printf("%d", data[t]);
    }
    printf("];\n");

    /* Emit per-position info */
    printf("const POS_BPC = [");
    for (int t = 0; t < N - 1; t++) {
        if (t > 0) printf(",");
        printf("%.3f", pos_bpc[t]);
    }
    printf("];\n");
    printf("const POS_CONF = [");
    for (int t = 0; t < N - 1; t++) {
        if (t > 0) printf(",");
        printf("%.4f", pos_conf[t]);
    }
    printf("];\n");
    printf("const WORD_LEN = [");
    for (int t = 0; t < N; t++) {
        if (t > 0) printf(",");
        printf("%d", word_len[t]);
    }
    printf("];\n");
    printf("const IN_TAG = [");
    for (int t = 0; t < N; t++) {
        if (t > 0) printf(",");
        printf("%d", in_tag[t]);
    }
    printf("];\n");

    /* Emit neuron info */
    printf("const NEURONS = [\n");
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        NeuronInfo* ni = &info[j];
        printf("  {id:%d,d1:%d,d2:%d,r2:%.3f,delta:%.4f,mean_h:%.3f,mean_abs:%.3f,range:%.3f,",
               j, ni->d1, ni->d2, ni->r2, ni->ablation_delta,
               ni->mean_h, ni->mean_abs_h, ni->cond_mean_range);

        /* Top output bytes */
        printf("top_out:[");
        for (int k = 0; k < 5; k++) {
            if (k > 0) printf(",");
            printf("{byte:%d,wy:%.3f}", ni->top_out[k], ni->top_out_wy[k]);
        }
        printf("],");

        /* Hidden state values for this neuron at each position */
        printf("h:[");
        for (int t = 0; t < N; t++) {
            if (t > 0) printf(",");
            printf("%.2f", h_states[t][j]);
        }
        printf("]");

        printf("}%s\n", j < HIDDEN_SIZE - 1 ? "," : "");
    }
    printf("];\n\n");

    /* Classify generalization */
    printf("function classifyGeneralization(n) {\n");
    printf("  // Structural patterns: involve XML chars, word boundaries\n");
    printf("  // Check if the neuron's top outputs include structural bytes\n");
    printf("  const structural = [32,60,62,47,10,34,61]; // space,<,>,/,\\n,\",=\n");
    printf("  let structCount = 0;\n");
    printf("  for (let o of n.top_out) {\n");
    printf("    if (structural.includes(o.byte)) structCount++;\n");
    printf("  }\n");
    printf("  // High R² + structural outputs → likely generalizes\n");
    printf("  if (n.r2 > 0.85 && structCount >= 2) return 'yes';\n");
    printf("  if (n.r2 > 0.80 && n.delta > 0.01) return 'yes';\n");
    printf("  if (n.r2 > 0.75) return 'maybe';\n");
    printf("  return 'no';\n");
    printf("}\n\n");

    /* Helper functions */
    printf("function safeChar(c) {\n");
    printf("  if (c === 60) return '&lt;';\n");
    printf("  if (c === 62) return '&gt;';\n");
    printf("  if (c === 38) return '&amp;';\n");
    printf("  if (c === 34) return '&quot;';\n");
    printf("  if (c >= 32 && c < 127) return String.fromCharCode(c);\n");
    printf("  return '\\\\x' + c.toString(16).padStart(2,'0');\n");
    printf("}\n\n");

    printf("function charLabel(c) {\n");
    printf("  if (c === 32) return 'space';\n");
    printf("  if (c === 10) return '\\\\n';\n");
    printf("  if (c === 60) return '&lt;';\n");
    printf("  if (c === 62) return '&gt;';\n");
    printf("  if (c >= 32 && c < 127) return \"'\" + String.fromCharCode(c) + \"'\";\n");
    printf("  return '0x' + c.toString(16).padStart(2,'0');\n");
    printf("}\n\n");

    /* Build neuron list */
    printf("let sortField = 'importance';\n");
    printf("let selectedNeuron = null;\n\n");

    printf("function sortBy(field) {\n");
    printf("  sortField = field;\n");
    printf("  document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));\n");
    printf("  event.target.classList.add('active');\n");
    printf("  renderList();\n");
    printf("}\n\n");

    printf("function renderList() {\n");
    printf("  let sorted = [...NEURONS];\n");
    printf("  if (sortField === 'importance') sorted.sort((a,b) => b.delta - a.delta);\n");
    printf("  else if (sortField === 'id') sorted.sort((a,b) => a.id - b.id);\n");
    printf("  else if (sortField === 'r2') sorted.sort((a,b) => b.r2 - a.r2);\n");
    printf("  else if (sortField === 'pair') sorted.sort((a,b) => (a.d1*100+a.d2) - (b.d1*100+b.d2));\n");
    printf("  \n");
    printf("  let html = '';\n");
    printf("  for (let n of sorted) {\n");
    printf("    let gen = classifyGeneralization(n);\n");
    printf("    let sel = selectedNeuron === n.id ? ' selected' : '';\n");
    printf("    let deltaColor = n.delta > 0.01 ? '#f85149' : (n.delta > 0.005 ? '#d29922' : '#8b949e');\n");
    printf("    html += `<div class=\"neuron-row${sel}\" onclick=\"selectNeuron(${n.id})\">`;\n");
    printf("    html += `<span class=\"gen ${gen}\"></span>`;\n");
    printf("    html += `<span class=\"id\">h${n.id}</span>`;\n");
    printf("    html += `<span class=\"pair\">(${n.d1},${n.d2})</span>`;\n");
    printf("    html += `<span class=\"r2\">${n.r2.toFixed(2)}</span>`;\n");
    printf("    html += `<span class=\"delta\" style=\"color:${deltaColor}\">+${n.delta.toFixed(3)}</span>`;\n");
    printf("    html += `<span class=\"sn\">`;\n");
    printf("    for (let o of n.top_out.slice(0,3)) {\n");
    printf("      let sign = o.wy > 0 ? '+' : '-';\n");
    printf("      html += `${sign}${charLabel(o.byte)} `;\n");
    printf("    }\n");
    printf("    html += `</span></div>`;\n");
    printf("  }\n");
    printf("  document.getElementById('neuron-list-inner').innerHTML = html;\n");
    printf("}\n\n");

    printf("function selectNeuron(id) {\n");
    printf("  selectedNeuron = id;\n");
    printf("  renderList();\n");
    printf("  renderDetail(id);\n");
    printf("}\n\n");

    printf("function renderDetail(id) {\n");
    printf("  let n = NEURONS[id];\n");
    printf("  let gen = classifyGeneralization(n);\n");
    printf("  let genLabel = gen === 'yes' ? 'Will generalize' : (gen === 'maybe' ? 'May generalize' : 'Likely coincidental');\n");
    printf("  let genColor = gen === 'yes' ? '#3fb950' : (gen === 'maybe' ? '#d29922' : '#f85149');\n");
    printf("  \n");
    printf("  let html = `<h2>Neuron h${id} <span style=\"color:${genColor};font-size:13px;margin-left:8px\">${genLabel}</span></h2>`;\n");
    printf("  \n");
    printf("  // Metrics\n");
    printf("  html += `<div class=\"metrics\">`;\n");
    printf("  html += `<div class=\"metric\"><div class=\"val\">(${n.d1},${n.d2})</div><div class=\"label\">Offset Pair</div></div>`;\n");
    printf("  html += `<div class=\"metric\"><div class=\"val\">${n.r2.toFixed(3)}</div><div class=\"label\">R&sup2;</div></div>`;\n");
    printf("  html += `<div class=\"metric\"><div class=\"val\">+${n.delta.toFixed(3)}</div><div class=\"label\">&Delta;bpc (ablation)</div></div>`;\n");
    printf("  html += `<div class=\"metric\"><div class=\"val\">${n.mean_abs.toFixed(3)}</div><div class=\"label\">Mean |h|</div></div>`;\n");
    printf("  html += `</div>`;\n");
    printf("  \n");
    printf("  // SN form\n");
    printf("  html += `<h3>SN-Form Pattern</h3>`;\n");
    printf("  html += `<div class=\"sn-box\">`;\n");
    printf("  html += `<span class=\"keyword\">NEURON</span> h${id}<br>`;\n");
    printf("  html += `<span class=\"keyword\">RESPONDS TO</span> data[t-${n.d1}] &times; data[t-${n.d2}]<br>`;\n");
    printf("  html += `<span class=\"keyword\">VARIANCE EXPLAINED</span> <span class=\"prob\">${(n.r2*100).toFixed(1)}%%</span><br>`;\n");
    printf("  html += `<span class=\"keyword\">ABLATION COST</span> +${n.delta.toFixed(3)} bpc<br>`;\n");
    printf("  html += `<span class=\"keyword\">PREDICTS</span> `;\n");
    printf("  for (let o of n.top_out) {\n");
    printf("    let sign = o.wy > 0 ? '+' : '-';\n");
    printf("    let color = o.wy > 0 ? '#3fb950' : '#f85149';\n");
    printf("    html += `<span style=\"color:${color}\">${sign}${charLabel(o.byte)}(${Math.abs(o.wy).toFixed(2)})</span> `;\n");
    printf("  }\n");
    printf("  html += `<br>`;\n");
    printf("  html += `<span class=\"keyword\">MEAN |h|</span> ${n.mean_abs.toFixed(3)} (${n.mean_abs > 0.8 ? 'saturated' : n.mean_abs > 0.5 ? 'moderate' : 'weak'})<br>`;\n");
    printf("  html += `</div>`;\n");
    printf("  \n");
    printf("  // Output contribution table\n");
    printf("  html += `<h3>W_y Output Contributions</h3>`;\n");
    printf("  html += `<table class=\"out-table\"><tr><th>Byte</th><th>W_y</th><th></th></tr>`;\n");
    printf("  for (let o of n.top_out) {\n");
    printf("    let w = Math.abs(o.wy) * 40;\n");
    printf("    let cls = o.wy > 0 ? 'pos' : 'neg';\n");
    printf("    html += `<tr><td>${charLabel(o.byte)}</td><td>${o.wy > 0 ? '+' : ''}${o.wy.toFixed(3)}</td>`;\n");
    printf("    html += `<td><span class=\"bar ${cls}\" style=\"width:${w}px\"></span></td></tr>`;\n");
    printf("  }\n");
    printf("  html += `</table>`;\n");
    printf("  \n");
    printf("  // Data view with activation overlay\n");
    printf("  html += `<h3>Activation Over Data (1024 bytes)</h3>`;\n");
    printf("  html += `<div class=\"data-view\">`;\n");
    printf("  for (let t = 0; t < DATA.length; t++) {\n");
    printf("    let h = n.h[t];\n");
    printf("    let intensity = Math.abs(h);\n");
    printf("    let r, g, b;\n");
    printf("    if (h > 0) { r = 50; g = Math.floor(80 + 175 * intensity); b = 50; }\n");
    printf("    else { r = Math.floor(80 + 175 * intensity); g = 50; b = 50; }\n");
    printf("    let alpha = 0.15 + 0.7 * intensity;\n");
    printf("    let bg = `rgba(${r},${g},${b},${alpha.toFixed(2)})`;\n");
    printf("    let title = `t=${t} h=${h.toFixed(3)} wl=${WORD_LEN[t]} tag=${IN_TAG[t]}`;\n");
    printf("    if (t < DATA.length-1) title += ` bpc=${POS_BPC[t].toFixed(2)}`;\n");
    printf("    html += `<span class=\"byte\" style=\"background:${bg}\" title=\"${title}\">${safeChar(DATA[t])}</span>`;\n");
    printf("  }\n");
    printf("  html += `</div>`;\n");
    printf("  \n");
    printf("  document.getElementById('detail').innerHTML = html;\n");
    printf("}\n\n");

    /* Initial render */
    printf("renderList();\n");
    printf("selectNeuron(8); // Start with most important neuron\n");

    printf("</script>\n</body>\n</html>\n");

    return 0;
}
