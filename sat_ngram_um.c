/*
 * sat_ngram_um: Direct n-gram UM on small dataset.
 *
 * Counts all n-grams up to a maximum length, uses them as UM patterns
 * to predict next byte. No hidden layer needed.
 *
 * This should match the saturated RNN's performance (0.079 bpc),
 * demonstrating that the RNN is encoding n-gram statistics at all depths
 * up to its effective BPTT limit.
 *
 * Usage: sat_ngram_um <datafile> [max_order]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Simple trie for n-gram counting.
 * Each node stores counts for what byte follows this context.
 */
typedef struct TrieNode {
    struct TrieNode* children[256];
    int counts[256];   /* counts[c] = how many times byte c follows this context */
    int total;         /* sum of counts */
} TrieNode;

TrieNode* trie_new(void) {
    TrieNode* n = calloc(1, sizeof(TrieNode));
    return n;
}

void trie_free(TrieNode* n) {
    if (!n) return;
    for (int i = 0; i < 256; i++)
        trie_free(n->children[i]);
    free(n);
}

/* Insert an n-gram context and its following byte */
void trie_insert(TrieNode* root, unsigned char* ctx, int ctx_len, unsigned char next) {
    TrieNode* node = root;
    for (int i = 0; i < ctx_len; i++) {
        if (!node->children[ctx[i]])
            node->children[ctx[i]] = trie_new();
        node = node->children[ctx[i]];
    }
    node->counts[next]++;
    node->total++;
}

/* Look up a context node */
TrieNode* trie_lookup(TrieNode* root, unsigned char* ctx, int ctx_len) {
    TrieNode* node = root;
    for (int i = 0; i < ctx_len; i++) {
        if (!node->children[ctx[i]]) return NULL;
        node = node->children[ctx[i]];
    }
    return node;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <datafile> [max_order]\n", argv[0]);
        return 1;
    }

    int max_order = 50;
    if (argc >= 3) max_order = atoi(argv[2]);

    /* Load data */
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END);
    int len = (int)ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* data = malloc(len);
    fread(data, 1, len, f);
    fclose(f);

    printf("=== N-gram UM Analysis ===\n");
    printf("Data: %d bytes\n", len);
    printf("Max order: %d\n\n", max_order);

    /* Build trie with all n-gram counts */
    printf("Building n-gram trie...\n");
    TrieNode* root = trie_new();

    for (int t = 0; t < len - 1; t++) {
        /* Insert contexts of length 0 (unigram) up to max_order-1.
         * Context of length ctx_len for predicting data[t+1] is
         * the ctx_len bytes ending at position t: data[t-ctx_len+1..t] */
        int max_ctx = t + 1; /* can't look back further than start */
        if (max_ctx > max_order) max_ctx = max_order;
        for (int ctx_len = 0; ctx_len <= max_ctx; ctx_len++) {
            unsigned char* ctx = data + t + 1 - ctx_len;
            trie_insert(root, ctx, ctx_len, data[t + 1]);
        }
    }

    /* Evaluate at each order: use longest matching context with backoff */
    printf("\n%-8s  %10s  %12s  %s\n", "order", "bpc", "contexts", "note");
    printf("%-8s  %10s  %12s  %s\n", "-----", "---", "--------", "----");

    /* Also evaluate cumulative (backoff from order N down to 1) */
    for (int order = 1; order <= max_order; order++) {
        double total_loss = 0.0;
        int n_predictions = 0;
        int contexts_used = 0;
        int backoff_used = 0;

        for (int t = 0; t < len - 1; t++) {
            unsigned char target = data[t + 1];
            double prob = 0;

            /* Try context lengths from order-1 down to 0 (backoff) */
            for (int ctx_len = order - 1; ctx_len >= 0; ctx_len--) {
                if (ctx_len > t) continue;
                unsigned char* ctx = data + t - ctx_len + 1;
                /* Actually context is the ctx_len bytes ending at position t */
                ctx = data + t + 1 - ctx_len;

                TrieNode* node = trie_lookup(root, ctx, ctx_len);
                if (node && node->total > 0 && node->counts[target] > 0) {
                    prob = (double)node->counts[target] / node->total;
                    if (ctx_len == order - 1) contexts_used++;
                    else backoff_used++;
                    break;
                }
            }

            if (prob <= 0) prob = 1.0 / 256; /* uniform fallback */
            total_loss += -log2(prob);
            n_predictions++;
        }

        double bpc = total_loss / n_predictions;

        char note[64] = "";
        if (order == 2) snprintf(note, sizeof(note), "bigram");
        else if (order == 3) snprintf(note, sizeof(note), "trigram");
        else if (bpc < 0.1) snprintf(note, sizeof(note), "<-- near RNN (0.079)");
        else if (bpc < 0.01) { snprintf(note, sizeof(note), "converged"); }

        printf("%-8d  %10.4f  %12d  %s\n", order, bpc, contexts_used, note);
        fflush(stdout);

        /* Stop early if we've converged */
        if (bpc < 0.001 && order > 5) {
            printf("  (converged, stopping)\n");
            break;
        }
    }

    /* Now do the detailed analysis: which n-gram lengths explain the gap */
    printf("\n=== Gap Analysis ===\n");
    printf("Where does each order help?\n\n");

    /* For each prediction, find the longest matching context that changes the prediction */
    int depth_helps[max_order + 1]; /* how many predictions improved by depth d */
    double depth_bits_saved[max_order + 1]; /* total bits saved by using depth d vs d-1 */
    memset(depth_helps, 0, sizeof(int) * (max_order + 1));
    memset(depth_bits_saved, 0, sizeof(double) * (max_order + 1));

    for (int t = 0; t < len - 1; t++) {
        unsigned char target = data[t + 1];
        double prev_prob = 0;
        int prev_depth = -1;

        for (int ctx_len = 0; ctx_len < max_order && ctx_len <= t; ctx_len++) {
            unsigned char* ctx = data + t + 1 - ctx_len;
            TrieNode* node = trie_lookup(root, ctx, ctx_len);
            if (node && node->total > 0 && node->counts[target] > 0) {
                double prob = (double)node->counts[target] / node->total;
                if (prob > prev_prob) {
                    double bits_before = (prev_prob > 0) ? -log2(prev_prob) : 8.0;
                    double bits_after = -log2(prob);
                    depth_helps[ctx_len]++;
                    depth_bits_saved[ctx_len] += bits_before - bits_after;
                    prev_prob = prob;
                    prev_depth = ctx_len;
                }
            }
        }
    }

    printf("%-8s  %10s  %12s  %12s\n", "depth", "helps", "bits saved", "avg saved");
    printf("%-8s  %10s  %12s  %12s\n", "-----", "-----", "----------", "---------");
    double cumulative_saved = 0;
    for (int d = 0; d <= max_order && d < len; d++) {
        if (depth_helps[d] == 0 && d > 10) {
            /* Check if all remaining are zero */
            int any_more = 0;
            for (int dd = d; dd <= max_order && dd < len; dd++)
                if (depth_helps[dd] > 0) { any_more = 1; break; }
            if (!any_more) break;
        }
        if (depth_helps[d] > 0) {
            cumulative_saved += depth_bits_saved[d];
            printf("%-8d  %10d  %12.2f  %12.4f  (cumul: %.2f bits)\n",
                   d, depth_helps[d], depth_bits_saved[d],
                   depth_bits_saved[d] / depth_helps[d],
                   cumulative_saved);
        }
    }

    /* Generalization analysis */
    printf("\n=== Generalization Analysis ===\n");
    printf("Which patterns would generalize vs overfit?\n\n");

    /* Count unique n-grams at each length */
    /* Rough heuristic: patterns appearing >= 4 times might generalize */
    for (int order = 2; order <= 10; order++) {
        int unique = 0, frequent = 0;
        int total_count = 0;
        for (int t = order - 1; t < len; t++) {
            /* Check if this n-gram is unique or repeated */
            unsigned char* ctx = data + t - order + 1;
            TrieNode* node = trie_lookup(root, ctx, order - 1);
            if (node && node->total > 0) {
                /* This context exists */
            }
        }
        /* Actually, let's count distinct contexts at each depth */
        /* Simpler: count via the data */
        /* Count distinct (order)-grams */
        /* Use a simple hash set */
        int seen[65536]; /* cheap hash */
        memset(seen, 0, sizeof(seen));
        int distinct = 0;
        for (int t = 0; t <= len - order; t++) {
            unsigned int h = 0;
            for (int k = 0; k < order; k++)
                h = h * 257 + data[t + k];
            h = h % 65536;
            if (!seen[h]) { seen[h] = 1; distinct++; }
        }
        printf("  %d-grams: ~%d distinct (of %d possible positions)\n",
               order, distinct, len - order + 1);
    }

    trie_free(root);
    free(data);
    return 0;
}
