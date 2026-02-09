/*
 * skip2_rnn.c — Find skip-2-gram patterns in the sat-rnn hidden state.
 *
 * For each surviving skip-2-gram pattern (xa@off_a, xb@off_b → y),
 * we run the RNN forward pass and record:
 *   1. Which hidden neurons change significantly when xa is input
 *   2. Which of those neurons' state persists until the output position
 *   3. Whether the persisted state contributes to predicting y via Wy
 *
 * This traces the information flow: xa → Wx → h → Wh^(off_a) → h → Wy → y
 * showing how the RNN compresses skip-patterns into its hidden state.
 *
 * Usage: skip2_rnn <data_file> <model_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256

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

void rnn_step(RNN* rnn, unsigned char x_t) {
    float h_new[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float sum = rnn->bh[i] + rnn->Wx[i][x_t];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            sum += rnn->Wh[i][j] * rnn->h[j];
        h_new[i] = tanhf(sum);
    }
    for (int i = 0; i < HIDDEN_SIZE; i++)
        rnn->h[i] = h_new[i];
}

void softmax(float* logits, float* probs, int n) {
    float maxv = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > maxv) maxv = logits[i];
    float sum = 0;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - maxv);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++)
        probs[i] /= sum;
}

void get_output(RNN* rnn, float* logits) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = rnn->by[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            sum += rnn->Wy[i][j] * rnn->h[j];
        logits[i] = sum;
    }
}

static char safe(int c) { return (c >= 32 && c < 127) ? c : '.'; }

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <data_file> <model_file>\n", argv[0]);
        return 1;
    }

    FILE* df = fopen(argv[1], "rb");
    if (!df) { perror("data"); return 1; }
    unsigned char data[2048];
    int len = fread(data, 1, 2048, df);
    fclose(df);

    RNN rnn;
    load_model(&rnn, argv[2]);
    printf("Data: %d bytes, Model: %s\n\n", len, argv[2]);

    /* === 1. Run full forward pass, recording hidden states === */
    printf("=== Running forward pass, recording hidden states ===\n\n");

    float h_states[2048][HIDDEN_SIZE];  /* hidden state at each position */
    float logits[OUTPUT_SIZE];
    float probs[OUTPUT_SIZE];

    memset(rnn.h, 0, sizeof(rnn.h));
    for (int t = 0; t < len; t++) {
        rnn_step(&rnn, data[t]);
        memcpy(h_states[t], rnn.h, sizeof(rnn.h));
    }

    /* === 2. For each skip-2-gram pattern, trace hidden state changes === */

    /* Analyze offset pair (1, 2) first — the strongest predictor */
    int offset_pairs[][2] = {{1, 2}, {1, 4}, {1, 8}};
    int n_pairs = 3;

    for (int op = 0; op < n_pairs; op++) {
        int oa = offset_pairs[op][0];
        int ob = offset_pairs[op][1];

        printf("========================================\n");
        printf("Skip-2-gram tracing: offset (%d, %d)\n", oa, ob);
        printf("========================================\n\n");

        /* Find the most common surviving patterns */
        /* Collect (xa, xb, y) triples with counts in both halves */
        typedef struct {
            unsigned char xa, xb, y;
            int count_first, count_second;
        } Pat;
        Pat pats[4096];
        int np = 0;

        for (int t = ob; t < len - 1; t++) {
            int xa = data[t - oa + 1];
            int xb = data[t - ob + 1];
            int y = data[t + 1];
            int half = (t < 1023) ? 0 : 1;

            int found = -1;
            for (int i = 0; i < np; i++) {
                if (pats[i].xa == xa && pats[i].xb == xb && pats[i].y == y) {
                    found = i; break;
                }
            }
            if (found >= 0) {
                if (half == 0) pats[found].count_first++;
                else pats[found].count_second++;
            } else if (np < 4096) {
                pats[np].xa = xa; pats[np].xb = xb; pats[np].y = y;
                pats[np].count_first = (half == 0) ? 1 : 0;
                pats[np].count_second = (half == 0) ? 0 : 1;
                np++;
            }
        }

        /* Sort by total count, filter to surviving patterns */
        for (int i = 0; i < np - 1; i++)
            for (int j = i + 1; j < np; j++) {
                int ci = pats[i].count_first + pats[i].count_second;
                int cj = pats[j].count_first + pats[j].count_second;
                if (cj > ci) {
                    Pat tmp = pats[i]; pats[i] = pats[j]; pats[j] = tmp;
                }
            }

        /* For top 10 surviving patterns, trace hidden state */
        printf("Pattern  xa  xb  ->  y    count  neurons_involved  Wy_contribution\n\n");

        int shown = 0;
        for (int pi = 0; pi < np && shown < 10; pi++) {
            if (pats[pi].count_first == 0 || pats[pi].count_second == 0)
                continue;

            int xa = pats[pi].xa;
            int xb = pats[pi].xb;
            int y = pats[pi].y;
            int total = pats[pi].count_first + pats[pi].count_second;

            /* Find all positions where this pattern occurs */
            /* For each occurrence, measure:
             * 1. h_delta when xb is input (at position t-ob+1)
             * 2. h state at output position t
             * 3. Wy[y] . h at output position (contribution to y) */

            double avg_h_delta[HIDDEN_SIZE] = {0};
            double avg_h_at_output[HIDDEN_SIZE] = {0};
            double avg_wy_contrib = 0;
            double avg_prob_y = 0;
            int n_occ = 0;

            for (int t = ob; t < len - 1; t++) {
                if (data[t - oa + 1] != xa) continue;
                if (data[t - ob + 1] != xb) continue;
                if (data[t + 1] != y) continue;

                /* h_delta when xb is input: h[t-ob+1] - h[t-ob] */
                int t_xb = t - ob + 1;
                if (t_xb < 1) continue;

                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    double delta = h_states[t_xb][j] - h_states[t_xb - 1][j];
                    avg_h_delta[j] += fabs(delta);
                    avg_h_at_output[j] += h_states[t][j];
                }

                /* Wy contribution to output y */
                double wy_y = rnn.by[y];
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    wy_y += rnn.Wy[y][j] * h_states[t][j];
                avg_wy_contrib += wy_y;

                /* Actual probability the model assigns to y */
                float logits_t[OUTPUT_SIZE];
                float probs_t[OUTPUT_SIZE];
                for (int i = 0; i < OUTPUT_SIZE; i++) {
                    logits_t[i] = rnn.by[i];
                    for (int j = 0; j < HIDDEN_SIZE; j++)
                        logits_t[i] += rnn.Wy[i][j] * h_states[t][j];
                }
                softmax(logits_t, probs_t, OUTPUT_SIZE);
                avg_prob_y += probs_t[y];

                n_occ++;
            }

            if (n_occ == 0) continue;

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                avg_h_delta[j] /= n_occ;
                avg_h_at_output[j] /= n_occ;
            }
            avg_wy_contrib /= n_occ;
            avg_prob_y /= n_occ;

            /* Count neurons with significant delta when xb input */
            int n_active = 0;
            int top_neurons[5] = {-1,-1,-1,-1,-1};
            double top_delta[5] = {0};

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                if (avg_h_delta[j] > 0.1) n_active++;
                /* Track top 5 */
                for (int k = 0; k < 5; k++) {
                    if (avg_h_delta[j] > top_delta[k]) {
                        for (int l = 4; l > k; l--) {
                            top_neurons[l] = top_neurons[l-1];
                            top_delta[l] = top_delta[l-1];
                        }
                        top_neurons[k] = j;
                        top_delta[k] = avg_h_delta[j];
                        break;
                    }
                }
            }

            double bpc_model = -log2(avg_prob_y);
            printf("'%c' '%c' -> '%c'  %3d   %3d neurons   Wy=%.2f  P(y)=%.4f  bpc=%.2f\n",
                   safe(xa), safe(xb), safe(y), total,
                   n_active, avg_wy_contrib, avg_prob_y, bpc_model);
            printf("  top neurons (by xb-input delta): ");
            for (int k = 0; k < 5 && top_neurons[k] >= 0; k++)
                printf("h%d(%.2f) ", top_neurons[k], top_delta[k]);
            printf("\n");

            shown++;
        }

        /* === 3. Per-neuron analysis: which neurons carry skip information? === */
        printf("\n--- Neuron information flow ---\n");
        printf("For each neuron: how much does it change at xb input,\n");
        printf("and how much does that change persist to the output?\n\n");

        /* Average over ALL positions where offset-pair patterns fire */
        double neuron_input_delta[HIDDEN_SIZE] = {0};
        double neuron_output_contrib[HIDDEN_SIZE] = {0};
        int n_total = 0;

        for (int t = ob; t < len - 1; t++) {
            int t_xb = t - ob + 1;
            if (t_xb < 1) continue;

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                neuron_input_delta[j] += fabs(h_states[t_xb][j] - h_states[t_xb - 1][j]);
                /* How much does this neuron contribute to the correct output? */
                int y = data[t + 1];
                neuron_output_contrib[j] += rnn.Wy[y][j] * h_states[t][j];
            }
            n_total++;
        }

        if (n_total > 0) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                neuron_input_delta[j] /= n_total;
                neuron_output_contrib[j] /= n_total;
            }
        }

        /* Find neurons that both respond to inputs and contribute to outputs */
        typedef struct { int idx; double delta; double contrib; double product; } NeuronScore;
        NeuronScore scores[HIDDEN_SIZE];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            scores[j].idx = j;
            scores[j].delta = neuron_input_delta[j];
            scores[j].contrib = neuron_output_contrib[j];
            scores[j].product = neuron_input_delta[j] * fabs(neuron_output_contrib[j]);
        }

        /* Sort by product (both responsive and contributive) */
        for (int i = 0; i < HIDDEN_SIZE - 1; i++)
            for (int j = i + 1; j < HIDDEN_SIZE; j++)
                if (scores[j].product > scores[i].product) {
                    NeuronScore tmp = scores[i];
                    scores[i] = scores[j];
                    scores[j] = tmp;
                }

        printf("neuron  avg_delta  avg_Wy_contrib  delta*|contrib|\n");
        for (int i = 0; i < 15; i++)
            printf("h%-4d   %.4f     %+.4f          %.4f\n",
                   scores[i].idx, scores[i].delta,
                   scores[i].contrib, scores[i].product);

        printf("\n");
    }

    /* === 4. W_hh eigenstructure: which hidden-to-hidden paths carry information? === */
    printf("========================================\n");
    printf("W_hh information carrying capacity\n");
    printf("========================================\n\n");

    /* For each neuron pair (i, j), Wh[i][j] determines how much
     * h_j at time t influences h_i at time t+1. Large |Wh[i][j]|
     * means information flows from j to i. */

    /* Find the strongest W_hh connections */
    typedef struct { int from; int to; float weight; } Connection;
    Connection conns[HIDDEN_SIZE * HIDDEN_SIZE];
    int nc = 0;
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            conns[nc].from = j;
            conns[nc].to = i;
            conns[nc].weight = rnn.Wh[i][j];
            nc++;
        }

    /* Sort by absolute weight */
    for (int i = 0; i < nc - 1; i++)
        for (int j = i + 1; j < nc; j++)
            if (fabsf(conns[j].weight) > fabsf(conns[i].weight)) {
                Connection tmp = conns[i];
                conns[i] = conns[j];
                conns[j] = tmp;
            }

    printf("Top 20 W_hh connections (by |weight|):\n");
    printf("h_from -> h_to   weight\n");
    for (int i = 0; i < 20; i++)
        printf("h%-4d -> h%-4d  %+.4f\n",
               conns[i].from, conns[i].to, conns[i].weight);

    /* W_hh spectral norm (rough indicator of information persistence) */
    /* Power iteration for largest singular value */
    float v[HIDDEN_SIZE], u[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) v[i] = 1.0f / sqrtf(HIDDEN_SIZE);

    float sigma = 0;
    for (int iter = 0; iter < 100; iter++) {
        /* u = Wh * v */
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            u[i] = 0;
            for (int j = 0; j < HIDDEN_SIZE; j++)
                u[i] += rnn.Wh[i][j] * v[j];
        }
        float norm = 0;
        for (int i = 0; i < HIDDEN_SIZE; i++) norm += u[i] * u[i];
        norm = sqrtf(norm);
        for (int i = 0; i < HIDDEN_SIZE; i++) u[i] /= norm;

        /* v = Wh^T * u */
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            v[j] = 0;
            for (int i = 0; i < HIDDEN_SIZE; i++)
                v[j] += rnn.Wh[i][j] * u[i];
        }
        float norm2 = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) norm2 += v[j] * v[j];
        norm2 = sqrtf(norm2);
        sigma = norm2;
        for (int j = 0; j < HIDDEN_SIZE; j++) v[j] /= norm2;
    }

    printf("\nW_hh spectral norm (largest singular value): %.4f\n", sigma);
    printf("  sigma < 1 → information decays per step\n");
    printf("  sigma > 1 → information can amplify (risk of chaos)\n");
    printf("  sigma ≈ 1 → information preserved (ideal for skip-patterns)\n");

    /* === 5. Information persistence: trace a specific pattern through time === */
    printf("\n========================================\n");
    printf("Information persistence trace\n");
    printf("========================================\n\n");

    printf("For the top surviving skip-2-gram, trace h-state change\n");
    printf("from the first input to the output position.\n\n");

    /* Use ' '' '→' ' pattern (most common) at offset (1,2) */
    /* Find first occurrence */
    for (int t = 2; t < len - 1; t++) {
        if (data[t] != ' ' || data[t-1] != ' ' || data[t+1] != ' ') continue;

        printf("Position %d: data[%d]=' ' data[%d]=' ' → data[%d]=' '\n",
               t, t-1, t, t+1);
        printf("Hidden state at each step:\n\n");

        printf("step  position  byte  h_norm   top_h_changes\n");
        for (int s = t - 2; s <= t; s++) {
            if (s < 0) continue;
            float norm = 0;
            for (int j = 0; j < HIDDEN_SIZE; j++)
                norm += h_states[s][j] * h_states[s][j];
            norm = sqrtf(norm);

            printf("%4d  %8d  '%c'   %.4f  ", s - (t-2), s, safe(data[s]), norm);

            /* Top 3 changing neurons */
            if (s > 0) {
                int top3[3] = {-1,-1,-1};
                float top3d[3] = {0};
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    float d = fabsf(h_states[s][j] - h_states[s-1][j]);
                    for (int k = 0; k < 3; k++) {
                        if (d > top3d[k]) {
                            for (int l = 2; l > k; l--) {
                                top3[l] = top3[l-1]; top3d[l] = top3d[l-1];
                            }
                            top3[k] = j; top3d[k] = d;
                            break;
                        }
                    }
                }
                for (int k = 0; k < 3; k++)
                    printf("h%d(%+.3f) ", top3[k],
                           h_states[s][top3[k]] - h_states[s-1][top3[k]]);
            }
            printf("\n");
        }

        /* Show Wy contribution to output ' ' from each neuron */
        printf("\nWy[' '] contributions from top neurons:\n");
        typedef struct { int j; float contrib; } WyC;
        WyC wyc[HIDDEN_SIZE];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            wyc[j].j = j;
            wyc[j].contrib = rnn.Wy[' '][j] * h_states[t][j];
        }
        for (int i = 0; i < HIDDEN_SIZE - 1; i++)
            for (int j = i + 1; j < HIDDEN_SIZE; j++)
                if (fabsf(wyc[j].contrib) > fabsf(wyc[i].contrib)) {
                    WyC tmp = wyc[i]; wyc[i] = wyc[j]; wyc[j] = tmp;
                }

        for (int i = 0; i < 10; i++)
            printf("  h%-3d: Wy=%.4f * h=%.4f = %+.4f\n",
                   wyc[i].j, rnn.Wy[' '][wyc[i].j],
                   h_states[t][wyc[i].j], wyc[i].contrib);

        break; /* just first occurrence */
    }

    return 0;
}
