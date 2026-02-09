/*
 * trace_example.c — Trace UM patterns through the trained RNN numerically.
 *
 * For specific positions in the data, show the complete computation:
 *   1. Active UM patterns (which bytes at which offsets predict what)
 *   2. W_x contribution from current byte
 *   3. W_h contribution from previous hidden state
 *   4. Pre-tanh sum, post-tanh activation for key neurons
 *   5. W_y contribution to output logits
 *   6. Final probability and prediction
 *
 * Goal: every RNN prediction explained via traceable data patterns,
 * followable with high-school arithmetic.
 *
 * Usage: trace_example <data_file> <model_file> [position]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256
#define MAX_DATA 1100

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

static char safe(int c) { return (c >= 32 && c < 127) ? c : '.'; }

static void show_byte(int b) {
    if (b >= 32 && b < 127)
        printf("'%c'", b);
    else
        printf("0x%02x", b);
}

/* UM bigram counts for pattern identification */
static int bg_count[50][256][256];
static int bg_total[50][256];

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <data_file> <model_file> [position]\n", argv[0]);
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

    /* Run forward pass, record all hidden states and pre-tanh values */
    float h_states[MAX_DATA][HIDDEN_SIZE];
    float pre_tanh[MAX_DATA][HIDDEN_SIZE];
    float wx_contrib[MAX_DATA][HIDDEN_SIZE];
    float wh_contrib[MAX_DATA][HIDDEN_SIZE];

    for (int t = 0; t < N; t++) {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            float wx = rnn.Wx[i][data[t]];
            float wh = 0;
            for (int j = 0; j < HIDDEN_SIZE; j++)
                wh += rnn.Wh[i][j] * rnn.h[j];
            float sum = rnn.bh[i] + wx + wh;

            wx_contrib[t][i] = wx;
            wh_contrib[t][i] = wh;
            pre_tanh[t][i] = sum;
            rnn.h[i] = tanhf(sum);  /* will be overwritten below */
        }
        /* Recompute properly (rnn_step modifies h in place) */
        float h_new[HIDDEN_SIZE];
        /* h was already updated above, but not correctly because we used
         * the partially updated h. Let me redo properly: */
        /* Actually the loop above used rnn.h from the PREVIOUS step for wh,
         * then overwrote rnn.h[i] in place. This means later neurons in the
         * same step see partially updated h. That's a bug. Let me fix. */
        break; /* Need to redo this properly */
    }

    /* Redo forward pass correctly */
    memset(rnn.h, 0, sizeof(rnn.h));
    for (int t = 0; t < N; t++) {
        float h_prev[HIDDEN_SIZE];
        memcpy(h_prev, rnn.h, sizeof(h_prev));

        for (int i = 0; i < HIDDEN_SIZE; i++) {
            float wx = rnn.Wx[i][data[t]];
            float wh = 0;
            for (int j = 0; j < HIDDEN_SIZE; j++)
                wh += rnn.Wh[i][j] * h_prev[j];
            float sum = rnn.bh[i] + wx + wh;

            wx_contrib[t][i] = wx;
            wh_contrib[t][i] = wh;
            pre_tanh[t][i] = sum;
        }

        /* Update h */
        for (int i = 0; i < HIDDEN_SIZE; i++)
            rnn.h[i] = tanhf(pre_tanh[t][i]);
        memcpy(h_states[t], rnn.h, sizeof(rnn.h));
    }

    /* Learn UM bigram counts */
    memset(bg_count, 0, sizeof(bg_count));
    memset(bg_total, 0, sizeof(bg_total));
    for (int d = 1; d <= 30; d++) {
        for (int t = d; t < N - 1; t++) {
            bg_count[d - 1][data[t - d]][data[t + 1]]++;
            bg_total[d - 1][data[t - d]]++;
        }
    }

    /* Marginal */
    int marginal[256] = {0};
    for (int t = 0; t < N; t++) marginal[data[t]]++;

    printf("Data: %d bytes, model loaded.\n\n", N);

    /* ===================================================================
     * Select positions to trace
     * =================================================================== */

    int positions[10];
    int n_pos = 0;

    if (argc >= 4) {
        positions[0] = atoi(argv[3]);
        n_pos = 1;
    } else {
        /* Pick interesting positions:
         * - After '<' (XML tag opening)
         * - After space (word boundary)
         * - After common letter (mid-word)
         * - A position with high prediction confidence
         * - A position with low prediction confidence
         */

        /* Find positions with highest and lowest prediction confidence */
        float best_conf = 0, worst_conf = 1;
        int best_t = 50, worst_t = 50;

        for (int t = 30; t < N - 1; t++) {
            float logits[OUTPUT_SIZE];
            for (int y = 0; y < OUTPUT_SIZE; y++) {
                float sum = rnn.by[y];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    sum += rnn.Wy[y][j] * h_states[t][j];
                logits[y] = sum;
            }
            float maxl = logits[0];
            for (int y = 1; y < OUTPUT_SIZE; y++)
                if (logits[y] > maxl) maxl = logits[y];
            float sum_exp = 0;
            for (int y = 0; y < OUTPUT_SIZE; y++)
                sum_exp += expf(logits[y] - maxl);
            float p_true = expf(logits[data[t + 1]] - maxl) / sum_exp;

            if (p_true > best_conf) { best_conf = p_true; best_t = t; }
            if (p_true < worst_conf && p_true > 0.001f) { worst_conf = p_true; worst_t = t; }
        }

        /* Find a post-'<' position */
        int after_lt = -1;
        for (int t = 30; t < N - 1; t++)
            if (data[t] == '<') { after_lt = t; break; }

        /* Find a post-space position */
        int after_sp = -1;
        for (int t = 40; t < N - 1; t++)
            if (data[t] == ' ') { after_sp = t; break; }

        /* Find mid-word position */
        int mid_word = -1;
        for (int t = 50; t < N - 1; t++)
            if (data[t] >= 'a' && data[t] <= 'z' &&
                data[t-1] >= 'a' && data[t-1] <= 'z' &&
                data[t+1] >= 'a' && data[t+1] <= 'z') { mid_word = t; break; }

        if (after_lt > 0) positions[n_pos++] = after_lt;
        if (after_sp > 0) positions[n_pos++] = after_sp;
        if (mid_word > 0) positions[n_pos++] = mid_word;
        positions[n_pos++] = best_t;
        positions[n_pos++] = worst_t;
    }

    /* ===================================================================
     * Trace each position
     * =================================================================== */

    for (int pi = 0; pi < n_pos; pi++) {
        int t = positions[pi];
        if (t < 1 || t >= N - 1) continue;

        printf("================================================================\n");
        printf("POSITION t=%d: data[t]=", t);
        show_byte(data[t]);
        printf("  target y=data[t+1]=");
        show_byte(data[t + 1]);
        printf("\n\n");

        /* Show context */
        printf("Context (20 chars before → current → next):\n  \"");
        int ctx_start = t - 20;
        if (ctx_start < 0) ctx_start = 0;
        for (int i = ctx_start; i <= t + 1 && i < N; i++) {
            if (i == t) printf("[");
            printf("%c", safe(data[i]));
            if (i == t) printf("]");
        }
        printf("\"\n\n");

        /* ---------------------------------------------------------------
         * Step 1: I^ℓ × O — Active UM patterns at this position
         * --------------------------------------------------------------- */
        printf("--- Step 1: Active UM patterns (I^ℓ × O) ---\n\n");
        printf("Which earlier bytes predict the output ");
        show_byte(data[t + 1]);
        printf("?\n\n");

        int y_true = data[t + 1];
        printf("offset  byte       P(y|x@d)  P(y)    log-ratio  count\n");

        int key_offsets[] = {1, 2, 3, 7, 8, 12, 20};
        for (int oi = 0; oi < 7; oi++) {
            int d = key_offsets[oi];
            if (t - d < 0) continue;
            int x = data[t - d];
            float p_y = (float)marginal[y_true] / N;
            float p_cond = 0;
            if (bg_total[d - 1][x] > 0)
                p_cond = (float)bg_count[d - 1][x][y_true] / bg_total[d - 1][x];

            float lr = (p_cond > 0 && p_y > 0) ? log2f(p_cond / p_y) : -99;
            char marker = (fabsf(lr) > 1.0f) ? '*' : ' ';

            printf("  %2d    ", d);
            show_byte(x);
            printf("%*s  %.3f     %.3f   %+.2f      %d/%d %c\n",
                   6 - (x >= 32 && x < 127 ? 3 : 4), "",
                   p_cond, p_y, lr,
                   bg_count[d - 1][x][y_true], bg_total[d - 1][x], marker);
        }

        /* Find the most informative offsets */
        printf("\nStrongest predictors (|log-ratio| > 1 bit):\n");
        for (int d = 1; d <= 20; d++) {
            if (t - d < 0) continue;
            int x = data[t - d];
            float p_y = (float)marginal[y_true] / N;
            float p_cond = 0;
            if (bg_total[d - 1][x] > 0)
                p_cond = (float)bg_count[d - 1][x][y_true] / bg_total[d - 1][x];
            if (p_cond == 0 || p_y == 0) continue;
            float lr = log2f(p_cond / p_y);
            if (fabsf(lr) > 1.0f) {
                printf("  data[t-%d]=", d);
                show_byte(x);
                printf(": P(");
                show_byte(y_true);
                printf("|");
                show_byte(x);
                printf(")=%.3f vs P(");
                show_byte(y_true);
                printf(")=%.3f, %+.1f bits\n",
                       p_cond, p_y, lr);
            }
        }

        /* ---------------------------------------------------------------
         * Step 2: I × H × O — The RNN computation at this position
         * --------------------------------------------------------------- */
        printf("\n--- Step 2: RNN computation (I × H × O) ---\n\n");

        /* Find the most important neurons (by |Wy contribution to correct output|) */
        typedef struct { int idx; float wy_contrib; float h_val; float wx; float wh; float pre; } NInfo;
        NInfo neurons[HIDDEN_SIZE];

        for (int j = 0; j < HIDDEN_SIZE; j++) {
            neurons[j].idx = j;
            neurons[j].h_val = h_states[t][j];
            neurons[j].wx = wx_contrib[t][j];
            neurons[j].wh = wh_contrib[t][j];
            neurons[j].pre = pre_tanh[t][j];
            neurons[j].wy_contrib = rnn.Wy[y_true][j] * h_states[t][j];
        }

        /* Sort by |wy_contrib| descending */
        for (int i = 0; i < HIDDEN_SIZE - 1; i++)
            for (int jj = i + 1; jj < HIDDEN_SIZE; jj++)
                if (fabsf(neurons[jj].wy_contrib) > fabsf(neurons[i].wy_contrib)) {
                    NInfo tmp = neurons[i]; neurons[i] = neurons[jj]; neurons[jj] = tmp;
                }

        printf("Top 10 neurons contributing to P(");
        show_byte(y_true);
        printf("):\n\n");
        printf("neuron   h[t]    W_x     W_h     bias    pre    h=tanh  W_y[y]*h\n");

        float total_wy_pos = 0, total_wy_neg = 0;
        for (int i = 0; i < 10; i++) {
            NInfo* n = &neurons[i];
            printf("h%-4d  %+.3f  %+.3f  %+.3f  %+.3f  %+.3f  %+.3f  %+.3f\n",
                   n->idx, n->h_val, n->wx, n->wh,
                   rnn.bh[n->idx], n->pre, n->h_val, n->wy_contrib);
        }

        for (int j = 0; j < HIDDEN_SIZE; j++) {
            float c = rnn.Wy[y_true][j] * h_states[t][j];
            if (c > 0) total_wy_pos += c;
            else total_wy_neg += c;
        }
        printf("\nTotal W_y contribution: positive=%+.3f, negative=%+.3f, sum=%+.3f\n",
               total_wy_pos, total_wy_neg, total_wy_pos + total_wy_neg);
        printf("Output bias by[");
        show_byte(y_true);
        printf("]=%+.3f\n", rnn.by[y_true]);

        /* Compute full output */
        float logits[OUTPUT_SIZE];
        for (int y = 0; y < OUTPUT_SIZE; y++) {
            float sum = rnn.by[y];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                sum += rnn.Wy[y][j] * h_states[t][j];
            logits[y] = sum;
        }

        float maxl = logits[0];
        for (int y = 1; y < OUTPUT_SIZE; y++)
            if (logits[y] > maxl) maxl = logits[y];
        float sum_exp = 0;
        float probs[OUTPUT_SIZE];
        for (int y = 0; y < OUTPUT_SIZE; y++) {
            probs[y] = expf(logits[y] - maxl);
            sum_exp += probs[y];
        }
        for (int y = 0; y < OUTPUT_SIZE; y++)
            probs[y] /= sum_exp;

        printf("\nlogit(");
        show_byte(y_true);
        printf(")=%+.3f, P(");
        show_byte(y_true);
        printf(")=%.4f (%.1f bits)\n",
               logits[y_true], probs[y_true],
               -log2f(probs[y_true] + 1e-10f));

        /* Top predictions */
        printf("\nTop 5 predictions:\n");
        int sorted[OUTPUT_SIZE];
        for (int y = 0; y < OUTPUT_SIZE; y++) sorted[y] = y;
        for (int i = 0; i < 5; i++)
            for (int j = i + 1; j < OUTPUT_SIZE; j++)
                if (probs[sorted[j]] > probs[sorted[i]]) {
                    int tmp = sorted[i]; sorted[i] = sorted[j]; sorted[j] = tmp;
                }
        for (int i = 0; i < 5; i++) {
            int y = sorted[i];
            char mark = (y == y_true) ? '<' : ' ';
            printf("  ");
            show_byte(y);
            printf(": %.4f (%.1f bits)%c\n", probs[y], -log2f(probs[y] + 1e-10f), mark);
        }

        /* ---------------------------------------------------------------
         * Step 3: (I × T)^ℓ × O — The interpretable decomposition
         * --------------------------------------------------------------- */
        printf("\n--- Step 3: Interpretable decomposition ---\n\n");

        /* Compute word_len and in_tag at this position */
        int wl = 0;
        for (int i = 1; i <= t; i++) {
            if (data[i - 1] == ' ' || data[i - 1] == '\n' ||
                data[i - 1] == '<' || data[i - 1] == '>')
                wl = 0;
            else
                wl = (wl < 15) ? wl + 1 : 15;
        }

        int in_tag = 0;
        for (int i = 0; i <= t; i++) {
            if (data[i] == '<') in_tag = 1;
            else if (data[i] == '>') in_tag = 0;
        }

        printf("State features: word_len=%d, in_tag=%d\n", wl, in_tag);
        printf("Current byte: ");
        show_byte(data[t]);
        printf("\n\n");

        printf("The RNN's prediction at this position is explained by:\n");
        printf("  Byte context: data[t]=");
        show_byte(data[t]);
        for (int d = 1; d <= 20; d++) {
            if (t - d < 0) continue;
            int x = data[t - d];
            float p_y = (float)marginal[y_true] / N;
            float p_cond = 0;
            if (bg_total[d - 1][x] > 0)
                p_cond = (float)bg_count[d - 1][x][y_true] / bg_total[d - 1][x];
            if (p_cond == 0 || p_y == 0) continue;
            float lr = log2f(p_cond / p_y);
            if (fabsf(lr) > 0.5f) {
                printf(", data[t-%d]=", d);
                show_byte(x);
                printf("(%+.1fb)", lr);
            }
        }
        printf("\n");
        printf("  State: word_len=%d, in_tag=%d\n", wl, in_tag);
        printf("  Combined prediction: P(");
        show_byte(y_true);
        printf(")=%.4f\n", probs[y_true]);

        /* Show the W_h source: for the top neurons, which neurons at t-1
         * contributed most to their W_h input */
        printf("\n--- W_h trace for top neuron ---\n\n");
        int top_j = neurons[0].idx;
        printf("Neuron h%d: h[t]=%.3f, W_y contrib=%.3f\n", top_j,
               neurons[0].h_val, neurons[0].wy_contrib);
        printf("  W_x[h%d][", top_j);
        show_byte(data[t]);
        printf("] = %+.3f (from current byte)\n", neurons[0].wx);
        printf("  bh[h%d] = %+.3f\n", top_j, rnn.bh[top_j]);
        printf("  W_h contribution = %+.3f (from previous hidden state)\n", neurons[0].wh);

        /* Which previous neurons contributed most to this W_h? */
        typedef struct { int idx; float contrib; } WH_source;
        WH_source wh_sources[HIDDEN_SIZE];
        for (int k = 0; k < HIDDEN_SIZE; k++) {
            wh_sources[k].idx = k;
            float h_prev = (t > 0) ? h_states[t - 1][k] : 0;
            wh_sources[k].contrib = rnn.Wh[top_j][k] * h_prev;
        }
        for (int i = 0; i < HIDDEN_SIZE - 1; i++)
            for (int j = i + 1; j < HIDDEN_SIZE; j++)
                if (fabsf(wh_sources[j].contrib) > fabsf(wh_sources[i].contrib)) {
                    WH_source tmp = wh_sources[i];
                    wh_sources[i] = wh_sources[j]; wh_sources[j] = tmp;
                }

        printf("  Top 5 W_h sources (from h[t-1]):\n");
        for (int i = 0; i < 5; i++) {
            WH_source* s = &wh_sources[i];
            float h_prev = (t > 0) ? h_states[t - 1][s->idx] : 0;
            printf("    h[t-1][%d]=%.3f × W_h[%d][%d]=%.3f → %+.3f\n",
                   s->idx, h_prev, top_j, s->idx,
                   rnn.Wh[top_j][s->idx], s->contrib);
        }

        printf("\n");
    }

    /* ===================================================================
     * Summary: overall bpc decomposition
     * =================================================================== */

    printf("================================================================\n");
    printf("OVERALL: bpc by position confidence\n\n");

    /* Compute bpc at each position */
    double total_loss = 0;
    int cnt = 0;
    for (int t = 0; t < N - 1; t++) {
        float logits_t[OUTPUT_SIZE];
        for (int y = 0; y < OUTPUT_SIZE; y++) {
            float sum = rnn.by[y];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                sum += rnn.Wy[y][j] * h_states[t][j];
            logits_t[y] = sum;
        }
        float maxl = logits_t[0];
        for (int y = 1; y < OUTPUT_SIZE; y++)
            if (logits_t[y] > maxl) maxl = logits_t[y];
        float sum_exp = 0;
        for (int y = 0; y < OUTPUT_SIZE; y++)
            sum_exp += expf(logits_t[y] - maxl);
        float lp = logits_t[data[t + 1]] - maxl - logf(sum_exp);
        total_loss -= lp;
        cnt++;
    }

    printf("Overall bpc: %.4f (%d positions)\n", total_loss / cnt / log(2.0), cnt);

    return 0;
}
