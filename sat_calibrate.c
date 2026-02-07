/*
 * sat_calibrate: Strength calibration between UM n-gram counts and RNN chains.
 *
 * For each n-gram (length 2..11, count >= 2):
 *   1. UM strength = log2(count)
 *   2. Find best matching RNN chain (trace through hidden activations)
 *   3. RNN chain strength = min(2|w_i|) over links
 *   4. Output TSV + summary statistics
 *
 * Usage: sat_calibrate <datafile> <model.bin> [output.tsv]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 256
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 256
#define MAX_ORDER 11
#define STRENGTH_THRESH 0.1f

typedef struct {
    float Wx[HIDDEN_SIZE][INPUT_SIZE];
    float Wh[HIDDEN_SIZE][HIDDEN_SIZE];
    float bh[HIDDEN_SIZE];
    float Wy[OUTPUT_SIZE][HIDDEN_SIZE];
    float by[OUTPUT_SIZE];
} RNN;

/* Trie for n-gram counting */
typedef struct TrieNode {
    struct TrieNode* children[256];
    int counts[256];
    int total;
} TrieNode;

TrieNode* trie_new(void) { return calloc(1, sizeof(TrieNode)); }

void trie_free(TrieNode* n) {
    if (!n) return;
    for (int i = 0; i < 256; i++) trie_free(n->children[i]);
    free(n);
}

void trie_insert(TrieNode* root, unsigned char* ctx, int ctx_len, unsigned char next) {
    TrieNode* node = root;
    for (int i = 0; i < ctx_len; i++) {
        if (!node->children[ctx[i]]) node->children[ctx[i]] = trie_new();
        node = node->children[ctx[i]];
    }
    node->counts[next]++;
    node->total++;
}

TrieNode* trie_lookup(TrieNode* root, unsigned char* ctx, int ctx_len) {
    TrieNode* node = root;
    for (int i = 0; i < ctx_len; i++) {
        if (!node->children[ctx[i]]) return NULL;
        node = node->children[ctx[i]];
    }
    return node;
}

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

char printable(unsigned char c) {
    return (c >= 32 && c < 127) ? c : '.';
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <datafile> <model.bin> [output.tsv]\n", argv[0]);
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

    /* Build trie */
    TrieNode* root = trie_new();
    for (int t = 0; t < len - 1; t++) {
        int max_ctx = t + 1;
        if (max_ctx > MAX_ORDER) max_ctx = MAX_ORDER;
        for (int ctx_len = 0; ctx_len <= max_ctx; ctx_len++) {
            unsigned char* ctx = data + t + 1 - ctx_len;
            trie_insert(root, ctx, ctx_len, data[t + 1]);
        }
    }

    /* Forward pass: record all hidden states */
    float (*hs)[HIDDEN_SIZE] = malloc(len * sizeof(float[HIDDEN_SIZE]));
    memset(hs[0], 0, sizeof(float) * HIDDEN_SIZE);

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
    }

    /* Open TSV output */
    const char* tsv_path = (argc >= 4) ? argv[3] : "calibration.tsv";
    FILE* tsv = fopen(tsv_path, "w");
    if (!tsv) { perror("tsv"); return 1; }
    fprintf(tsv, "ngram\tlength\tcount\tum_strength\trnn_strength\tratio\n");

    /* Per-length statistics */
    int n_per_len[MAX_ORDER + 1] = {0};
    double sum_um[MAX_ORDER + 1] = {0};
    double sum_rnn[MAX_ORDER + 1] = {0};
    double sum_ratio[MAX_ORDER + 1] = {0};
    int n_matched[MAX_ORDER + 1] = {0};

    /* Track which n-grams we've already processed (dedup) */
    /* Simple: use position of first occurrence */

    printf("=== Strength Calibration ===\n");
    printf("Data: %d bytes, Model: %s\n\n", len, argv[2]);

    for (int order = 2; order <= MAX_ORDER; order++) {
        for (int t = order - 2; t < len - 1; t++) {
            int ctx_len = order - 1;
            if (ctx_len > t) continue;
            unsigned char* ctx = data + t + 1 - ctx_len;
            unsigned char target = data[t + 1];

            /* Check if this n-gram has been processed already */
            int ngram_start = t + 1 - ctx_len;
            int is_first = 1;
            for (int tt = 0; tt < ngram_start; tt++) {
                if (tt + order > len) break;
                if (memcmp(data + tt, data + ngram_start, order) == 0) {
                    is_first = 0;
                    break;
                }
            }
            if (!is_first) continue;

            /* Get UM count */
            TrieNode* node = trie_lookup(root, ctx, ctx_len);
            if (!node || node->counts[target] < 2) continue;
            int count = node->counts[target];
            double um_s = log2(count);

            /* Trace RNN chain for this n-gram.
             * We need to find the chain through the actual hidden states
             * at the positions where this n-gram occurs. Use position t. */
            int chain_start = t + 1 - ctx_len;  /* position of first char */
            int chain_end = t + 1;               /* position of target char */

            /* The chain goes: input(data[chain_start]) ->Wx-> h[chain_start+1]
             * ->Wh-> h[chain_start+2] ->...->Wh-> h[chain_end] ->Wy-> output(target) */

            /* Find best chain:
             * At each step, find the hidden neuron that contributes most */
            double best_chain_strength = 0;

            /* Try all possible starting neurons */
            for (int h0 = 0; h0 < HIDDEN_SIZE; h0++) {
                float wx_s = 2.0f * rnn.Wx[h0][data[chain_start]];
                if (wx_s <= STRENGTH_THRESH) continue;
                if (hs[chain_start + 1][h0] <= 0) continue;  /* must be active */

                /* Greedily trace through recurrence */
                float min_strength = wx_s;
                int cur_h = h0;
                int broken = 0;

                for (int step = chain_start + 1; step < chain_end; step++) {
                    /* Find best next neuron from cur_h */
                    int best_next = -1;
                    float best_wh = 0;
                    for (int h1 = 0; h1 < HIDDEN_SIZE; h1++) {
                        if (hs[step + 1][h1] <= 0) continue;
                        float wh_s = 2.0f * rnn.Wh[h1][cur_h];
                        if (wh_s > best_wh) {
                            best_wh = wh_s;
                            best_next = h1;
                        }
                    }
                    if (best_next < 0 || best_wh <= STRENGTH_THRESH) {
                        broken = 1;
                        break;
                    }
                    if (best_wh < min_strength) min_strength = best_wh;
                    cur_h = best_next;
                }

                if (broken) continue;

                /* Check output link */
                float wy_s = 2.0f * rnn.Wy[target][cur_h];
                if (wy_s <= STRENGTH_THRESH) continue;
                if (wy_s < min_strength) min_strength = wy_s;

                if (min_strength > best_chain_strength)
                    best_chain_strength = min_strength;
            }

            /* Write TSV line */
            char ngram_str[MAX_ORDER + 1];
            for (int i = 0; i < order; i++)
                ngram_str[i] = printable(data[ngram_start + i]);
            ngram_str[order] = '\0';

            double ratio = (best_chain_strength > 0) ? best_chain_strength / um_s : 0;

            fprintf(tsv, "%s\t%d\t%d\t%.3f\t%.3f\t%.3f\n",
                    ngram_str, order, count, um_s, best_chain_strength, ratio);

            n_per_len[order]++;
            sum_um[order] += um_s;
            if (best_chain_strength > 0) {
                sum_rnn[order] += best_chain_strength;
                sum_ratio[order] += ratio;
                n_matched[order]++;
            }
        }
    }

    fclose(tsv);
    printf("Wrote %s\n\n", tsv_path);

    /* Summary */
    printf("=== Summary by N-gram Length ===\n\n");
    printf("Len  N-grams  Matched  Avg-UM   Avg-RNN  Avg-Ratio  Match%%\n");
    for (int order = 2; order <= MAX_ORDER; order++) {
        if (n_per_len[order] == 0) continue;
        double avg_um = sum_um[order] / n_per_len[order];
        double avg_rnn = (n_matched[order] > 0) ? sum_rnn[order] / n_matched[order] : 0;
        double avg_ratio = (n_matched[order] > 0) ? sum_ratio[order] / n_matched[order] : 0;
        double match_pct = 100.0 * n_matched[order] / n_per_len[order];
        printf("%3d  %7d  %7d  %7.3f  %7.3f  %9.3f  %5.1f%%\n",
               order, n_per_len[order], n_matched[order],
               avg_um, avg_rnn, avg_ratio, match_pct);
    }

    printf("\n=== Overall ===\n");
    int total_ngrams = 0, total_matched = 0;
    double total_um = 0, total_rnn = 0, total_ratio = 0;
    for (int order = 2; order <= MAX_ORDER; order++) {
        total_ngrams += n_per_len[order];
        total_matched += n_matched[order];
        total_um += sum_um[order];
        total_rnn += sum_rnn[order];
        total_ratio += sum_ratio[order];
    }
    printf("Total n-grams (count >= 2): %d\n", total_ngrams);
    printf("Matched with RNN chain: %d (%.1f%%)\n",
           total_matched, 100.0 * total_matched / total_ngrams);
    if (total_matched > 0)
        printf("Overall avg ratio: %.3f\n", total_ratio / total_matched);

    free(hs);
    free(data);
    trie_free(root);
    return 0;
}
