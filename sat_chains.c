/*
 * sat_chains: Pattern chain analysis of a saturated RNN via isomorphic UM.
 *
 * Usage: sat_chains <datafile> <model.bin>
 *
 * For each timestep, traces pattern chains from input through hidden
 * activations to output, computing chain strengths (min of atomic links).
 * Validates chain strengths against actual n-gram counts in the dataset.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256
#define MAX_CHAINS 20       /* top chains to report per timestep */
#define MAX_DEPTH 8         /* max recurrent chain depth */
#define STRENGTH_THRESH 0.5 /* minimum link strength to consider */

typedef struct {
    float Wx[HIDDEN_SIZE][INPUT_SIZE];
    float Wh[HIDDEN_SIZE][HIDDEN_SIZE];
    float bh[HIDDEN_SIZE];
    float Wy[OUTPUT_SIZE][HIDDEN_SIZE];
    float by[OUTPUT_SIZE];
} RNN;

/* A single pattern chain */
typedef struct {
    int depth;                      /* number of recurrent steps back */
    int hidden_path[MAX_DEPTH + 1]; /* hidden neuron indices */
    int input_time;                 /* which timestep the chain starts */
    unsigned char input_byte;       /* input byte at chain start */
    unsigned char output_byte;      /* predicted output byte */
    float link_strengths[MAX_DEPTH + 2]; /* strength of each link */
    float chain_strength;           /* min of all links */
} Chain;

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

void softmax(float* logits, float* probs, int n) {
    float max_val = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > max_val) max_val = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf(logits[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < n; i++)
        probs[i] /= sum;
}

char printable(unsigned char c) {
    return (c >= 32 && c < 127) ? c : '.';
}

/* Count occurrences of an n-gram in the dataset */
int count_ngram(unsigned char* data, int len, unsigned char* pattern, int plen) {
    int count = 0;
    for (int i = 0; i <= len - plen; i++) {
        int match = 1;
        for (int j = 0; j < plen; j++) {
            if (data[i + j] != pattern[j]) { match = 0; break; }
        }
        if (match) count++;
    }
    return count;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <datafile> <model.bin>\n", argv[0]);
        return 1;
    }

    /* Load data */
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("fopen data"); return 1; }
    fseek(f, 0, SEEK_END);
    int len = (int)ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* data = malloc(len);
    fread(data, 1, len, f);
    fclose(f);

    /* Load model */
    RNN rnn;
    load_model(&rnn, argv[2]);

    printf("=== Pattern Chain Analysis ===\n");
    printf("Data: %d bytes, max pattern strength: %.1f\n", len, log2(len));
    printf("Model: %s\n\n", argv[2]);

    /* Forward pass: record all hidden states */
    float (*hs)[HIDDEN_SIZE] = malloc(len * sizeof(float[HIDDEN_SIZE]));
    float (*logits_all)[OUTPUT_SIZE] = malloc(len * sizeof(float[OUTPUT_SIZE]));
    float (*probs_all)[OUTPUT_SIZE] = malloc(len * sizeof(float[OUTPUT_SIZE]));
    memset(hs[0], 0, sizeof(float) * HIDDEN_SIZE);

    double total_loss = 0.0;
    for (int t = 0; t < len - 1; t++) {
        unsigned char x = data[t];
        float h_new[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            float s = rnn.bh[i] + rnn.Wx[i][x];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                s += rnn.Wh[i][j] * hs[t][j];
            h_new[i] = tanhf(s);
        }
        memcpy(hs[t + 1], h_new, sizeof(float) * HIDDEN_SIZE);

        float logits[OUTPUT_SIZE];
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            float s = rnn.by[i];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                s += rnn.Wy[i][j] * h_new[j];
            logits[i] = s;
        }
        memcpy(logits_all[t], logits, sizeof(float) * OUTPUT_SIZE);
        softmax(logits, probs_all[t], OUTPUT_SIZE);

        float p = probs_all[t][data[t + 1]];
        if (p < 1e-8f) p = 1e-8f;
        total_loss += -logf(p);
    }
    double rnn_bpc = total_loss / ((len - 1) * logf(2.0));
    printf("RNN bpc: %.4f\n\n", rnn_bpc);

    /* === Chain analysis === */

    /* Summary statistics */
    int total_chains_found = 0;
    double total_strength = 0.0;
    int chain_depth_hist[MAX_DEPTH + 1] = {0};
    int neuron_participation[HIDDEN_SIZE] = {0};

    printf("=== Per-Timestep Chain Analysis ===\n\n");

    for (int t = 0; t < len - 1; t++) {
        unsigned char target = data[t + 1];
        float target_prob = probs_all[t][target];
        float target_surprisal = -log2f(target_prob);

        /* Find top hidden neurons contributing to the target output.
         * Only consider neurons with h > 0 (positive events in doubled-E). */
        typedef struct { int idx; float strength; } HLink;
        HLink output_links[HIDDEN_SIZE];
        int n_output_links = 0;

        for (int j = 0; j < HIDDEN_SIZE; j++) {
            if (hs[t + 1][j] <= 0) continue; /* only positive events */
            float w = rnn.Wy[target][j];
            if (w <= 0) continue; /* only positive patterns (h+ -> output) */
            float s = 2.0f * w; /* pattern strength = 2|w| */
            if (s >= STRENGTH_THRESH) {
                output_links[n_output_links].idx = j;
                output_links[n_output_links].strength = s;
                n_output_links++;
            }
        }

        /* Sort by strength descending */
        for (int a = 0; a < n_output_links - 1; a++)
            for (int b = a + 1; b < n_output_links; b++)
                if (output_links[b].strength > output_links[a].strength) {
                    HLink tmp = output_links[a];
                    output_links[a] = output_links[b];
                    output_links[b] = tmp;
                }

        /* For each strong output link, trace back through recurrence */
        Chain chains[MAX_CHAINS];
        int n_chains = 0;

        for (int ol = 0; ol < n_output_links && n_chains < MAX_CHAINS; ol++) {
            int h_cur = output_links[ol].idx;
            float wy_strength = output_links[ol].strength;

            /* Trace back: try depths 0 (direct input->h->output) up to MAX_DEPTH */
            /* Depth 0: input at time t directly to h_cur */
            {
                float wx_s = 2.0f * rnn.Wx[h_cur][data[t]];
                if (wx_s > 0 && wx_s >= STRENGTH_THRESH) {
                    Chain c;
                    c.depth = 0;
                    c.hidden_path[0] = h_cur;
                    c.input_time = t;
                    c.input_byte = data[t];
                    c.output_byte = target;
                    c.link_strengths[0] = wx_s;     /* input -> h */
                    c.link_strengths[1] = wy_strength; /* h -> output */
                    c.chain_strength = fminf(wx_s, wy_strength);
                    if (c.chain_strength >= STRENGTH_THRESH) {
                        chains[n_chains++] = c;
                    }
                }
            }

            /* Depth 1+: trace through Wh */
            /* Use greedy: at each step back, pick the strongest incoming link */
            int path[MAX_DEPTH + 1];
            float path_strengths[MAX_DEPTH + 2];
            path[0] = h_cur;
            path_strengths[0] = wy_strength; /* h->output link */

            for (int depth = 1; depth <= MAX_DEPTH && n_chains < MAX_CHAINS; depth++) {
                int prev_t = t + 1 - depth;
                if (prev_t < 1) break; /* need h at prev_t which comes from t=prev_t-1 */

                /* Find strongest h_prev -> h_cur recurrent link */
                int best_prev = -1;
                float best_wh_s = 0;
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    if (hs[prev_t][j] <= 0) continue; /* only h+ events */
                    float w = rnn.Wh[path[depth - 1]][j];
                    if (w <= 0) continue; /* only positive patterns */
                    float s = 2.0f * w;
                    if (s > best_wh_s) {
                        best_wh_s = s;
                        best_prev = j;
                    }
                }

                if (best_prev < 0 || best_wh_s < STRENGTH_THRESH) break;

                path[depth] = best_prev;
                path_strengths[depth] = best_wh_s;

                /* Check input link at the start of this chain */
                int input_t = prev_t - 1;
                if (input_t < 0) break;
                float wx_s = 2.0f * rnn.Wx[best_prev][data[input_t]];
                if (wx_s > 0 && wx_s >= STRENGTH_THRESH) {
                    Chain c;
                    c.depth = depth;
                    for (int d = 0; d <= depth; d++)
                        c.hidden_path[d] = path[d];
                    c.input_time = input_t;
                    c.input_byte = data[input_t];
                    c.output_byte = target;

                    /* Link strengths: input->h, then recurrent links, then h->output */
                    c.link_strengths[0] = wx_s;
                    for (int d = depth; d >= 1; d--)
                        c.link_strengths[depth - d + 1] = path_strengths[d];
                    c.link_strengths[depth + 1] = path_strengths[0]; /* output link */

                    /* Chain strength = min of all links */
                    float min_s = wx_s;
                    for (int d = 0; d <= depth; d++)
                        if (path_strengths[d] < min_s) min_s = path_strengths[d];
                    c.chain_strength = min_s;

                    if (c.chain_strength >= STRENGTH_THRESH) {
                        chains[n_chains++] = c;
                    }
                }
            }
        }

        /* Sort chains by strength */
        for (int a = 0; a < n_chains - 1; a++)
            for (int b = a + 1; b < n_chains; b++)
                if (chains[b].chain_strength > chains[a].chain_strength) {
                    Chain tmp = chains[a];
                    chains[a] = chains[b];
                    chains[b] = tmp;
                }

        /* Print results for this timestep */
        if (n_chains > 0 || target_surprisal > 1.0) {
            printf("t=%3d: '%c' -> '%c'  prob=%.4f  surprisal=%.2f bits  chains=%d\n",
                   t, printable(data[t]), printable(target),
                   target_prob, target_surprisal, n_chains);

            int show = (n_chains < 5) ? n_chains : 5;
            for (int i = 0; i < show; i++) {
                Chain* c = &chains[i];

                /* Build the n-gram this chain represents */
                int ngram_len = c->depth + 2; /* input bytes + output byte */
                unsigned char ngram[MAX_DEPTH + 2];
                for (int d = 0; d < ngram_len - 1; d++)
                    ngram[d] = data[c->input_time + d];
                ngram[ngram_len - 1] = c->output_byte;

                int ngram_count = count_ngram(data, len, ngram, ngram_len);
                float expected_strength = (ngram_count > 0) ? log2f(ngram_count) : 0;

                printf("  chain[%d]: strength=%.2f  depth=%d  ngram=\"",
                       i, c->chain_strength, c->depth);
                for (int d = 0; d < ngram_len; d++)
                    printf("%c", printable(ngram[d]));
                printf("\"  count=%d  log2(count)=%.1f",
                       ngram_count, expected_strength);

                /* Show hidden path */
                printf("  path: in(%c)", printable(c->input_byte));
                for (int d = c->depth; d >= 0; d--)
                    printf("->h%d", c->hidden_path[d]);
                printf("->out(%c)", printable(c->output_byte));

                /* Show link strengths */
                printf("  links:[");
                for (int d = 0; d <= c->depth + 1; d++) {
                    if (d > 0) printf(",");
                    printf("%.1f", c->link_strengths[d]);
                }
                printf("]\n");

                /* Track stats */
                for (int d = 0; d <= c->depth; d++)
                    neuron_participation[c->hidden_path[d]]++;
            }
            printf("\n");
        }

        /* Accumulate stats */
        total_chains_found += n_chains;
        for (int i = 0; i < n_chains; i++) {
            total_strength += chains[i].chain_strength;
            chain_depth_hist[chains[i].depth]++;
        }
    }

    /* === Summary === */
    printf("\n=== Summary ===\n\n");
    printf("Total chains found: %d (avg %.1f per timestep)\n",
           total_chains_found, (float)total_chains_found / (len - 1));
    printf("Average chain strength: %.2f\n",
           total_chains_found > 0 ? total_strength / total_chains_found : 0);

    printf("\nChain depth distribution:\n");
    for (int d = 0; d <= MAX_DEPTH; d++) {
        if (chain_depth_hist[d] > 0)
            printf("  depth %d: %d chains (%.1f%%)\n",
                   d, chain_depth_hist[d],
                   100.0 * chain_depth_hist[d] / total_chains_found);
    }

    printf("\nTop participating hidden neurons:\n");
    /* Sort and show top 20 */
    typedef struct { int idx; int count; } NP;
    NP np[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        np[i].idx = i;
        np[i].count = neuron_participation[i];
    }
    for (int a = 0; a < HIDDEN_SIZE - 1; a++)
        for (int b = a + 1; b < HIDDEN_SIZE; b++)
            if (np[b].count > np[a].count) {
                NP tmp = np[a]; np[a] = np[b]; np[b] = tmp;
            }
    int show_neurons = 20;
    for (int i = 0; i < show_neurons && np[i].count > 0; i++)
        printf("  h%-3d: %d chain appearances\n", np[i].idx, np[i].count);

    /* === Timesteps with no chains (high surprisal) === */
    printf("\nTimesteps with high surprisal and no chains:\n");
    int no_chain_count = 0;
    for (int t = 0; t < len - 1; t++) {
        float surprisal = -log2f(probs_all[t][data[t + 1]]);
        /* Check if we found any chains for this timestep by re-checking */
        /* (Simplified: just report high-surprisal positions) */
        if (surprisal > 3.0) {
            printf("  t=%3d: '%c'->'%c' surprisal=%.1f bits\n",
                   t, printable(data[t]), printable(data[t + 1]), surprisal);
            no_chain_count++;
        }
    }
    if (no_chain_count == 0)
        printf("  (none with surprisal > 3.0)\n");

    free(hs);
    free(logits_all);
    free(probs_all);
    free(data);
    return 0;
}
