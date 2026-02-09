/*
 * skip_kgram.c - General skip-k-gram analysis for the pattern-chain UM
 *
 * Measures H(Y | X@d1, X@d2, ..., X@dk) for arbitrary offset tuples.
 * Builds the diminishing-returns curve: how many non-contiguous bytes
 * are needed to match the contiguous n-gram model?
 *
 * Usage: ./skip_kgram <datafile>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_DATA 2048
#define MAX_K 8
#define NBYTES 256

static unsigned char data[MAX_DATA];
static int data_len;

/* Hash table for k-gram contexts */
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

/*
 * H(Y | X@offsets[0], ..., X@offsets[k-1])
 * offsets are distances before the output position
 */
static double skip_kgram_bpc(const int *offsets, int k, int *n_patterns) {
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
    int patterns = 0;
    for (int i = 0; i < HASH_SIZE; i++) {
        for (Entry *e = table[i]; e; e = e->next) {
            for (int y = 0; y < NBYTES; y++) {
                if (e->count[y] == 0) continue;
                patterns++;
                double p = (double)e->count[y] / e->total;
                H -= (double)e->count[y] / N * log2(p);
            }
        }
    }

    if (n_patterns) *n_patterns = patterns;
    return H;
}

/* Greedy offset selection: given current offsets, find the next offset
 * that minimizes H(Y | current_offsets, new_offset) */
static int greedy_next(const int *offsets, int k, int max_try, double *best_h) {
    int best_off = -1;
    *best_h = 999;

    int new_offsets[MAX_K];
    memcpy(new_offsets, offsets, k * sizeof(int));

    for (int d = 1; d <= max_try; d++) {
        /* Skip if already in set */
        int dup = 0;
        for (int i = 0; i < k; i++)
            if (offsets[i] == d) { dup = 1; break; }
        if (dup) continue;

        new_offsets[k] = d;
        double h = skip_kgram_bpc(new_offsets, k + 1, NULL);
        if (h < *best_h) {
            *best_h = h;
            best_off = d;
        }
    }
    return best_off;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <datafile>\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    data_len = fread(data, 1, MAX_DATA, f);
    fclose(f);

    printf("Data: %d bytes\n\n", data_len);

    /* Marginal */
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

    /* Contiguous n-gram baselines */
    printf("=== Contiguous n-gram baselines ===\n");
    printf("order  offsets                 bpc      patterns\n");
    for (int order = 1; order <= 8; order++) {
        int offsets[MAX_K];
        for (int i = 0; i < order; i++) offsets[i] = i + 1;
        int np;
        double h = skip_kgram_bpc(offsets, order, &np);
        printf("  %d    [", order);
        for (int i = 0; i < order; i++) printf("%s%d", i?",":"", offsets[i]);
        printf("]%*s  %.4f    %d\n", 22 - 2*order, "", h, np);
    }

    /* Greedy skip-gram: start with offset 1, greedily add best offset */
    printf("\n=== Greedy skip-k-gram (start with offset 1, add best next) ===\n");
    printf("k  offsets                 bpc      patterns  vs contiguous-k\n");

    int offsets[MAX_K] = {1};
    int k = 1;
    int np;
    double h = skip_kgram_bpc(offsets, 1, &np);

    /* Contiguous baselines for comparison */
    double contig_bpc[MAX_K+1];
    for (int order = 1; order <= MAX_K; order++) {
        int co[MAX_K];
        for (int i = 0; i < order; i++) co[i] = i + 1;
        contig_bpc[order] = skip_kgram_bpc(co, order, NULL);
    }

    printf("  1  [1]                    %.4f    %d      (baseline)\n", h, np);

    for (k = 1; k < MAX_K; k++) {
        double best_h;
        int best_off = greedy_next(offsets, k, 30, &best_h);
        if (best_off < 0) break;
        offsets[k] = best_off;

        int new_np;
        skip_kgram_bpc(offsets, k+1, &new_np);

        printf("  %d  [", k+1);
        for (int i = 0; i <= k; i++) printf("%s%d", i?",":"", offsets[i]);
        printf("]%*s  %.4f    %d      contig-%d = %.4f\n",
               22 - 2*(k+1), "", best_h, new_np, k+1, contig_bpc[k+1]);
    }

    /* Alternative: greedy from scratch (no fixed start) */
    printf("\n=== Greedy skip-k-gram (no fixed start) ===\n");
    printf("k  offsets                 bpc      patterns\n");

    /* Start by finding the single best offset */
    int offsets2[MAX_K];
    double best_single = 999;
    int best_single_off = -1;
    for (int d = 1; d <= 30; d++) {
        int o[1] = {d};
        double hh = skip_kgram_bpc(o, 1, NULL);
        if (hh < best_single) { best_single = hh; best_single_off = d; }
    }
    offsets2[0] = best_single_off;
    printf("  1  [%d]                    %.4f\n", best_single_off, best_single);

    for (int kk = 1; kk < MAX_K; kk++) {
        double best_h;
        int best_off = greedy_next(offsets2, kk, 30, &best_h);
        if (best_off < 0) break;
        offsets2[kk] = best_off;

        int new_np;
        skip_kgram_bpc(offsets2, kk+1, &new_np);

        printf("  %d  [", kk+1);
        for (int i = 0; i <= kk; i++) printf("%s%d", i?",":"", offsets2[i]);
        printf("]%*s  %.4f    %d\n", 22 - 2*(kk+1), "", best_h, new_np);
    }

    /* Show specific interesting patterns from the best skip-3 */
    printf("\n=== Top patterns from best greedy-3 offsets ===\n");
    {
        int best3[3];
        memcpy(best3, offsets, 3 * sizeof(int));
        clear_table();

        int max_off = 0;
        for (int i = 0; i < 3; i++)
            if (best3[i] > max_off) max_off = best3[i];

        int N = 0;
        for (int t = max_off; t < data_len; t++) {
            unsigned char ctx[3];
            for (int i = 0; i < 3; i++)
                ctx[i] = data[t - best3[i]];
            int y = data[t];

            unsigned int hh = hash_ctx(ctx, 3);
            Entry *e = table[hh];
            while (e) {
                int match = 1;
                for (int i = 0; i < 3; i++)
                    if (e->ctx[i] != ctx[i]) { match = 0; break; }
                if (match) break;
                e = e->next;
            }
            if (!e) {
                e = calloc(1, sizeof(Entry));
                memcpy(e->ctx, ctx, 3);
                e->next = table[hh];
                table[hh] = e;
            }
            e->count[y]++;
            e->total++;
            N++;
        }

        /* Collect and sort by count */
        typedef struct { unsigned char ctx[3]; int y; int count; int total; } Pat;
        Pat *pats = malloc(100000 * sizeof(Pat));
        int npats = 0;

        for (int i = 0; i < HASH_SIZE; i++) {
            for (Entry *e = table[i]; e; e = e->next) {
                for (int y = 0; y < NBYTES; y++) {
                    if (e->count[y] == 0) continue;
                    pats[npats].ctx[0] = e->ctx[0];
                    pats[npats].ctx[1] = e->ctx[1];
                    pats[npats].ctx[2] = e->ctx[2];
                    pats[npats].y = y;
                    pats[npats].count = e->count[y];
                    pats[npats].total = e->total;
                    npats++;
                }
            }
        }

        /* Sort by count descending */
        for (int i = 0; i < npats-1; i++)
            for (int j = i+1; j < npats; j++)
                if (pats[j].count > pats[i].count) {
                    Pat tmp = pats[i]; pats[i] = pats[j]; pats[j] = tmp;
                }

        char safe(int c) { return (c >= 32 && c < 127) ? c : '.'; }

        printf("Offsets: [%d, %d, %d]\n", best3[0], best3[1], best3[2]);
        printf("x@%d  x@%d  x@%d  ->  y     count  P(y|ctx)\n",
               best3[0], best3[1], best3[2]);

        int show = npats < 30 ? npats : 30;
        for (int i = 0; i < show; i++) {
            printf("'%c'   '%c'   '%c'   ->  '%c'     %3d    %.3f\n",
                   safe(pats[i].ctx[0]), safe(pats[i].ctx[1]), safe(pats[i].ctx[2]),
                   safe(pats[i].y),
                   pats[i].count, (double)pats[i].count / pats[i].total);
        }
        printf("Total patterns: %d\n", npats);
        free(pats);
    }

    return 0;
}
