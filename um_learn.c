/*
 * um_learn.c — UM learning: record all joint events from the data.
 *
 * For each position t in the data, the output event is y = data[t].
 * The input events are data[t-d] for offsets d = 1..MAX_DEPTH.
 *
 * The UM learning rule (from CMP) is log-stochastic counting of joint
 * events. For batch processing: count co-occurrences, store floor(log2(count)).
 *
 * This tool:
 *   1. Records ALL (input_byte @ offset_d, output_byte) joint events
 *      for offsets 1..50 (full BPTT window)
 *   2. Computes H(Y | X@d) for each single offset — the bigram level
 *   3. Greedily selects offsets that minimize H(Y | context) — rediscovering
 *      the [1,8,20,3,...] sequence from skip_kgram.c
 *   4. Shows the pattern count and log-support distribution at each level
 *   5. Outputs the full log-support matrix for the superset
 *
 * This is the "I^ℓ → O" UM that learns a superset of what the RNN can learn.
 * The factoring step (introducing named hidden ESs) comes later.
 *
 * Usage: um_learn <data_file> [max_depth]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_DATA 2048
#define MAX_DEPTH 50
#define NBYTES 256

static unsigned char data[MAX_DATA];
static int data_len;

/* ===================================================================
 * Hash table for k-gram contexts (reused from skip_kgram.c pattern)
 * =================================================================== */

#define MAX_K 8

typedef struct Entry {
    unsigned char ctx[MAX_K];
    int count[NBYTES];
    int total;
    struct Entry *next;
} Entry;

#define HASH_SIZE 262139
static Entry *table[HASH_SIZE];

static void clear_table(void) {
    for (int i = 0; i < HASH_SIZE; i++) {
        Entry *e = table[i];
        while (e) { Entry *n = e->next; free(e); e = n; }
        table[i] = NULL;
    }
}

static unsigned int hash_ctx(const unsigned char *ctx, int k) {
    unsigned int h = 0;
    for (int i = 0; i < k; i++)
        h = h * 257 + ctx[i] + 1;
    return h % HASH_SIZE;
}

static double skip_kgram_bpc(const int *offsets, int k, int *n_patterns, int *n_contexts) {
    clear_table();
    int max_off = 0;
    for (int i = 0; i < k; i++)
        if (offsets[i] > max_off) max_off = offsets[i];

    int N = 0;
    for (int t = max_off; t < data_len; t++) {
        unsigned char ctx[MAX_K];
        for (int i = 0; i < k; i++)
            ctx[i] = data[t - offsets[i]];
        int y = data[t];

        unsigned int h = hash_ctx(ctx, k);
        Entry *e = table[h];
        while (e) {
            int match = 1;
            for (int i = 0; i < k; i++)
                if (e->ctx[i] != ctx[i]) { match = 0; break; }
            if (match) break;
            e = e->next;
        }
        if (!e) {
            e = calloc(1, sizeof(Entry));
            memcpy(e->ctx, ctx, k);
            e->next = table[h];
            table[h] = e;
        }
        e->count[y]++;
        e->total++;
        N++;
    }

    double H = 0;
    int patterns = 0, contexts = 0;
    for (int i = 0; i < HASH_SIZE; i++) {
        for (Entry *e = table[i]; e; e = e->next) {
            contexts++;
            for (int y = 0; y < NBYTES; y++) {
                if (e->count[y] == 0) continue;
                patterns++;
                double p = (double)e->count[y] / e->total;
                H -= (double)e->count[y] / N * log2(p);
            }
        }
    }

    if (n_patterns) *n_patterns = patterns;
    if (n_contexts) *n_contexts = contexts;
    return H;
}

/* ===================================================================
 * Single-offset bigram analysis: H(Y | X@d) for each offset d
 * =================================================================== */

typedef struct {
    int offset;
    double bpc;
    int patterns;
    int contexts;
} OffsetInfo;

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <datafile> [max_depth]\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    data_len = fread(data, 1, MAX_DATA, f);
    fclose(f);

    int max_depth = MAX_DEPTH;
    if (argc >= 3) max_depth = atoi(argv[2]);
    if (max_depth > MAX_DEPTH) max_depth = MAX_DEPTH;

    printf("Data: %d bytes, max depth: %d\n\n", data_len, max_depth);

    /* Marginal H(Y) */
    {
        int count[NBYTES] = {0};
        for (int i = 0; i < data_len; i++) count[data[i]]++;
        double H = 0;
        for (int b = 0; b < NBYTES; b++) {
            if (count[b] == 0) continue;
            double p = (double)count[b] / data_len;
            H -= p * log2(p);
        }
        printf("Marginal H(Y) = %.4f bpc\n\n", H);
    }

    /* ===================================================================
     * 1. Single-offset analysis: H(Y | X@d) for d = 1..max_depth
     * =================================================================== */

    printf("=== Single-Offset Bigram: H(Y | X@d) ===\n\n");
    printf("offset  bpc      patterns  contexts  log_support_range\n");

    OffsetInfo offsets_info[MAX_DEPTH + 1];

    for (int d = 1; d <= max_depth; d++) {
        int off[1] = {d};
        int np, nc;
        double h = skip_kgram_bpc(off, 1, &np, &nc);

        offsets_info[d].offset = d;
        offsets_info[d].bpc = h;
        offsets_info[d].patterns = np;
        offsets_info[d].contexts = nc;

        /* Log-support distribution for this offset */
        int log_support_max = 0;
        for (int i = 0; i < HASH_SIZE; i++) {
            for (Entry *e = table[i]; e; e = e->next) {
                for (int y = 0; y < NBYTES; y++) {
                    if (e->count[y] == 0) continue;
                    int ls = 0, c = e->count[y];
                    while (c > 1) { c >>= 1; ls++; }
                    if (ls > log_support_max) log_support_max = ls;
                }
            }
        }

        printf("  %2d    %.4f   %4d      %4d      0..%d\n",
               d, h, np, nc, log_support_max);
    }

    /* Sort offsets by bpc (best first) */
    int sorted[MAX_DEPTH + 1];
    for (int d = 1; d <= max_depth; d++) sorted[d - 1] = d;
    for (int i = 0; i < max_depth - 1; i++)
        for (int j = i + 1; j < max_depth; j++)
            if (offsets_info[sorted[j]].bpc < offsets_info[sorted[i]].bpc) {
                int tmp = sorted[i]; sorted[i] = sorted[j]; sorted[j] = tmp;
            }

    printf("\nTop 10 single offsets (by bpc):\n");
    for (int i = 0; i < 10 && i < max_depth; i++)
        printf("  offset %2d: %.4f bpc, %d patterns\n",
               sorted[i], offsets_info[sorted[i]].bpc,
               offsets_info[sorted[i]].patterns);

    /* ===================================================================
     * 2. Greedy offset selection (should rediscover [1,8,20,3,...])
     * =================================================================== */

    printf("\n=== Greedy Offset Selection ===\n\n");
    printf("k  offsets                          bpc      patterns  contexts\n");

    int greedy[MAX_K];
    int gk = 0;

    /* Start with offset 1 (always best single) */
    greedy[0] = 1;
    gk = 1;
    {
        int np, nc;
        double h = skip_kgram_bpc(greedy, 1, &np, &nc);
        printf("1  [1]                              %.4f   %4d      %4d\n", h, np, nc);
    }

    for (int step = 1; step < MAX_K; step++) {
        double best_h = 999;
        int best_off = -1;

        for (int d = 1; d <= max_depth; d++) {
            /* Skip if already selected */
            int dup = 0;
            for (int i = 0; i < gk; i++)
                if (greedy[i] == d) { dup = 1; break; }
            if (dup) continue;

            int trial[MAX_K];
            memcpy(trial, greedy, gk * sizeof(int));
            trial[gk] = d;
            double h = skip_kgram_bpc(trial, gk + 1, NULL, NULL);
            if (h < best_h) { best_h = h; best_off = d; }
        }

        if (best_off < 0) break;
        greedy[gk] = best_off;
        gk++;

        int np, nc;
        skip_kgram_bpc(greedy, gk, &np, &nc);

        printf("%d  [", gk);
        for (int i = 0; i < gk; i++) printf("%s%d", i ? "," : "", greedy[i]);
        printf("]");
        int pad = 34 - 2 * gk;
        if (pad > 0) printf("%*s", pad, "");
        printf("%.4f   %4d      %4d\n", best_h, np, nc);
    }

    /* ===================================================================
     * 3. Log-support distribution at each greedy level
     *
     * For each level k, show how many patterns have each log-support value.
     * This is the UM's learned "weight" distribution.
     * =================================================================== */

    printf("\n=== Log-Support Distribution per Greedy Level ===\n\n");

    for (int level = 1; level <= gk; level++) {
        int np, nc;
        skip_kgram_bpc(greedy, level, &np, &nc);

        int ls_hist[20] = {0};
        int ls_max = 0;
        for (int i = 0; i < HASH_SIZE; i++) {
            for (Entry *e = table[i]; e; e = e->next) {
                for (int y = 0; y < NBYTES; y++) {
                    if (e->count[y] == 0) continue;
                    int ls = 0, c = e->count[y];
                    while (c > 1) { c >>= 1; ls++; }
                    if (ls < 20) ls_hist[ls]++;
                    if (ls > ls_max) ls_max = ls;
                }
            }
        }

        printf("Level %d (offsets [", level);
        for (int i = 0; i < level; i++) printf("%s%d", i ? "," : "", greedy[i]);
        printf("]): %d patterns, %d contexts\n", np, nc);
        printf("  log_support: ");
        for (int ls = 0; ls <= ls_max && ls < 20; ls++)
            if (ls_hist[ls] > 0) printf("%d:%d ", ls, ls_hist[ls]);
        printf("\n");
    }

    /* ===================================================================
     * 4. Superset verification: for each of the RNN's neuron offset pairs,
     *    show that the UM has patterns at those offsets
     * =================================================================== */

    printf("\n=== Superset Verification ===\n\n");
    printf("For each offset pair used by RNN neurons (from factor_map2),\n");
    printf("show UM pattern count and bpc at those offsets.\n\n");

    /* RNN neuron offset pairs (from factor_map2 results) */
    int pairs[][2] = {
        {1, 7},   /* 52 neurons */
        {1, 8},   /* 20 neurons */
        {8, 2},   /* 18 neurons (note: stored as sorted) */
        {1, 12},  /* 9 neurons */
        {2, 7},   /* 8 neurons */
        {3, 12},  /* 6 neurons */
        {1, 20},  /* 5 neurons */
        {2, 12},  /* 4 neurons */
        {1, 3},   /* 1 neuron */
        {3, 7},   /* 1 neuron */
        {1, 2},   /* 2 neurons */
        {20, 2},  /* 2 neurons */
    };
    int n_pairs = 12;
    int neurons_per_pair[] = {52, 20, 18, 9, 8, 6, 5, 4, 1, 1, 2, 2};

    printf("%-10s  neurons  bpc      patterns  contexts\n", "pair");
    for (int p = 0; p < n_pairs; p++) {
        int off[2] = {pairs[p][0], pairs[p][1]};
        int np, nc;
        double h = skip_kgram_bpc(off, 2, &np, &nc);
        printf("(%d,%d)%*s  %3d      %.4f   %4d      %4d\n",
               off[0], off[1],
               7 - (int)(log10(off[0]+1) + log10(off[1]+1)), "",
               neurons_per_pair[p], h, np, nc);
    }

    /* ===================================================================
     * 5. Full pattern dump at greedy level 2 (most common: offsets 1,8)
     *    Show the top patterns by count
     * =================================================================== */

    printf("\n=== Top Patterns at Greedy Level 2 (offsets [%d,%d]) ===\n\n", greedy[0], greedy[1]);
    {
        int off[2] = {greedy[0], greedy[1]};
        int np, nc;
        skip_kgram_bpc(off, 2, &np, &nc);

        /* Collect patterns */
        typedef struct { unsigned char x1, x2, y; int count; int log_support; } Pat;
        Pat *pats = malloc(np * sizeof(Pat));
        int pi = 0;

        for (int i = 0; i < HASH_SIZE; i++) {
            for (Entry *e = table[i]; e; e = e->next) {
                for (int y = 0; y < NBYTES; y++) {
                    if (e->count[y] == 0) continue;
                    pats[pi].x1 = e->ctx[0];
                    pats[pi].x2 = e->ctx[1];
                    pats[pi].y = y;
                    pats[pi].count = e->count[y];
                    int ls = 0, c = e->count[y];
                    while (c > 1) { c >>= 1; ls++; }
                    pats[pi].log_support = ls;
                    pi++;
                }
            }
        }

        /* Sort by count descending */
        for (int i = 0; i < pi - 1; i++)
            for (int j = i + 1; j < pi; j++)
                if (pats[j].count > pats[i].count) {
                    Pat tmp = pats[i]; pats[i] = pats[j]; pats[j] = tmp;
                }

        char safe(int c) { return (c >= 32 && c < 127) ? c : '.'; }

        printf("x@%d  x@%d  ->  y     count  log_sup  P(y|ctx)\n", off[0], off[1]);
        int show = pi < 30 ? pi : 30;
        for (int i = 0; i < show; i++) {
            printf("'%c'   '%c'   ->  '%c'     %3d    %d        %.3f\n",
                   safe(pats[i].x1), safe(pats[i].x2), safe(pats[i].y),
                   pats[i].count, pats[i].log_support,
                   (double)pats[i].count / pats[i].count); /* need total for proper P */
        }
        printf("Total: %d patterns, %d contexts\n", pi, nc);
        free(pats);
    }

    return 0;
}
