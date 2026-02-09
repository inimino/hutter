/*
 * skip2gram.c — Skip-2-gram analysis of the 1024-byte dataset.
 *
 * A skip-2-gram is a pattern: x_a at offset a, x_b at offset b → output y
 * where offset a < offset b (both before the output position).
 *
 * We enumerate these patterns, compute how much bpc each explains
 * (improvement over the best single-offset prediction), and rank them.
 *
 * With DSS=1024, there's almost no generalization — these are pure data-terms.
 * When we double to DSS=2048, patterns that recur are real; those that don't
 * were artifacts of this specific dataset.
 *
 * Usage: skip2gram <data_file> [max_offset]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_DATA 2048
#define MAX_OFFSET 50   /* BPTT limit — no point going further for RNN comparison */
#define N_BYTES 256

/* A skip-2-gram pattern: two input bytes at two offsets predict an output */
typedef struct {
    int x_a;        /* first input byte */
    int off_a;      /* first offset (closer to output) */
    int x_b;        /* second input byte */
    int off_b;      /* second offset (further from output) */
    int y;          /* output byte */
    int count;      /* how many times this pattern occurs */
    double bpc_gain; /* bpc improvement over best single-offset */
} Skip2Gram;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_file> [max_offset]\n", argv[0]);
        return 1;
    }

    int max_off = MAX_OFFSET;
    if (argc > 2) max_off = atoi(argv[2]);
    if (max_off > MAX_OFFSET) max_off = MAX_OFFSET;

    FILE* df = fopen(argv[1], "rb");
    if (!df) { perror("data"); return 1; }
    unsigned char data[MAX_DATA];
    int len = fread(data, 1, sizeof(data), df);
    fclose(df);
    printf("Data: %d bytes, max offset: %d\n\n", len, max_off);

    /* === 0. Marginal output distribution === */
    int out_count[N_BYTES] = {0};
    int n_outputs = len - 1;
    for (int t = 0; t < n_outputs; t++)
        out_count[data[t + 1]]++;

    double h_out = 0;
    for (int y = 0; y < N_BYTES; y++) {
        if (out_count[y] == 0) continue;
        double p = (double)out_count[y] / n_outputs;
        h_out -= p * log2(p);
    }
    printf("H(output) = %.4f bits (marginal bpc = %.4f)\n\n", h_out, h_out);

    /* === 1. Single-offset baseline: H(Y|X@offset) for each offset === */
    printf("=== Single-offset baselines ===\n");
    printf("offset  H(Y|X)   MI(X;Y)  bpc\n");

    double h_y_given_x_best[MAX_OFFSET + 1];
    for (int off = 1; off <= max_off && off < len; off++) {
        int cx[N_BYTES] = {0};
        int joint[N_BYTES][N_BYTES];
        memset(joint, 0, sizeof(joint));
        int n = 0;

        for (int t = off; t < len - 1; t++) {
            int x = data[t - off + 1];  /* input at offset 'off' before output */
            int y = data[t + 1];        /* output */
            cx[x]++;
            joint[x][y]++;
            n++;
        }

        double h_cond = 0;
        for (int x = 0; x < N_BYTES; x++) {
            if (cx[x] == 0) continue;
            double px = (double)cx[x] / n;
            double hx = 0;
            for (int y = 0; y < N_BYTES; y++) {
                if (joint[x][y] == 0) continue;
                double py_x = (double)joint[x][y] / cx[x];
                hx -= py_x * log2(py_x);
            }
            h_cond += px * hx;
        }

        h_y_given_x_best[off] = h_cond;
        double mi = h_out - h_cond;
        if (off <= 15)
            printf("%6d  %6.3f   %6.3f   %.3f\n", off, h_cond, mi, h_cond);
    }

    /* === 2. Skip-2-gram: for each offset pair (a,b), compute H(Y|X_a,X_b) === */
    printf("\n=== Skip-2-gram: offset pair analysis ===\n");
    printf("Conditional entropy H(Y | X@a, X@b) for top offset pairs\n");
    printf("Improvement = min(H(Y|X@a), H(Y|X@b)) - H(Y|X@a,X@b)\n\n");

    /* Store results for all offset pairs */
    typedef struct {
        int a, b;
        double h_joint;
        double improvement;
        int n_patterns;  /* distinct (x_a, x_b, y) triples with count > 0 */
    } OffsetPair;

    int max_pairs = (max_off * (max_off - 1)) / 2;
    OffsetPair* pairs = calloc(max_pairs, sizeof(OffsetPair));
    int n_pairs = 0;

    /* We need to be memory-efficient. For each offset pair, we need counts
     * of (x_a, x_b) and (x_a, x_b, y). With 256^3 = 16M entries per pair
     * that's too much. Instead, since DSS=1024, we have at most ~1000 positions,
     * so we can just iterate and use a hash-based approach. */

    /* Actually with 1024 positions, we can use a direct array for the joint
     * distribution of (x_a, x_b) — that's only 256*256=64K entries. Then
     * for each (x_a, x_b) bucket we track which y values appear. */

    /* For the full (x_a, x_b, y) we'd need 256^3 which is 16M — fine as ints. */
    /* But we only have ~1000 data points so most will be 0. Use sparse approach. */

    /* Sparse approach: for each offset pair, collect all (x_a, x_b, y) triples,
     * then compute conditional entropy directly. */

    printf("off_a off_b  H(Y|Xa,Xb)  improve  #patterns  best_single\n");

    /* Only check offset pairs up to 15 for the detailed table (225 pairs) */
    int detail_max = (max_off < 15) ? max_off : 15;

    for (int a = 1; a <= detail_max && a < len; a++) {
        for (int b = a + 1; b <= detail_max && b < len; b++) {
            /* Count joint distribution */
            /* Key: a*256 + xa gives the (offset, byte) for first input */
            /* We need P(y | x_a, x_b) */

            /* Use a flat array: index = xa * 256 + xb, value = array of y counts */
            /* Since we only have ~1000 positions, use a list approach */

            int n = 0;
            int n_pats = 0;

            /* pair_key = xa * 256 + xb (64K possible) */
            /* For each pair_key, track count and per-y counts */
            /* Use two arrays: pair_count[64K] and pair_y_count[64K][256] */
            /* 64K * 256 * 4 = 64MB — too much for stack, use malloc */

            /* Better: since we have ~1000 data points, just collect them
             * and compute entropy directly */

            /* Collect all (xa, xb, y) triples */
            typedef struct { unsigned char xa, xb, y; } Triple;
            Triple triples[MAX_DATA];
            int nt = 0;

            int min_start = b;  /* need at least b positions before output */
            for (int t = min_start; t < len - 1; t++) {
                int xa = data[t - a + 1];
                int xb = data[t - b + 1];
                int y = data[t + 1];
                triples[nt].xa = xa;
                triples[nt].xb = xb;
                triples[nt].y = y;
                nt++;
            }
            n = nt;

            /* Compute H(Y | Xa, Xb) using the triples */
            /* Group by (xa, xb), compute H(Y) within each group */
            /* Use a hash map: key = xa*256+xb */
            int pair_count[65536];
            memset(pair_count, 0, sizeof(pair_count));
            /* First pass: count pairs */
            for (int i = 0; i < nt; i++) {
                int key = triples[i].xa * 256 + triples[i].xb;
                pair_count[key]++;
            }

            /* For each non-zero pair, compute H(Y|Xa=xa,Xb=xb) */
            /* We need y counts per pair. Do it pair by pair. */
            double h_cond = 0;
            int counted_pairs = 0;

            for (int key = 0; key < 65536; key++) {
                if (pair_count[key] == 0) continue;
                int xa = key / 256;
                int xb = key % 256;

                /* Count y values for this pair */
                int yc[N_BYTES] = {0};
                int ny = 0;
                for (int i = 0; i < nt; i++) {
                    if (triples[i].xa == xa && triples[i].xb == xb) {
                        yc[triples[i].y]++;
                    }
                }

                double pk = (double)pair_count[key] / n;
                double hk = 0;
                for (int y = 0; y < N_BYTES; y++) {
                    if (yc[y] == 0) continue;
                    n_pats++;
                    double py = (double)yc[y] / pair_count[key];
                    hk -= py * log2(py);
                }
                h_cond += pk * hk;
            }

            double best_single = h_y_given_x_best[a];
            if (h_y_given_x_best[b] < best_single)
                best_single = h_y_given_x_best[b];

            double improvement = best_single - h_cond;

            if (n_pairs < max_pairs) {
                pairs[n_pairs].a = a;
                pairs[n_pairs].b = b;
                pairs[n_pairs].h_joint = h_cond;
                pairs[n_pairs].improvement = improvement;
                pairs[n_pairs].n_patterns = n_pats;
                n_pairs++;
            }

            if (improvement > 0.1)  /* only print interesting pairs */
                printf("%5d %5d  %10.4f  %7.4f  %9d  %.4f\n",
                       a, b, h_cond, improvement, n_pats, best_single);
        }
    }

    /* === 3. Top offset pairs by improvement === */
    printf("\n=== Top 20 offset pairs by bpc improvement ===\n\n");
    printf("off_a off_b  H(Y|Xa,Xb)  improve  #patterns\n");

    /* Sort by improvement */
    for (int i = 0; i < n_pairs - 1; i++)
        for (int j = i + 1; j < n_pairs; j++)
            if (pairs[j].improvement > pairs[i].improvement) {
                OffsetPair tmp = pairs[i];
                pairs[i] = pairs[j];
                pairs[j] = tmp;
            }

    for (int i = 0; i < 20 && i < n_pairs; i++)
        printf("%5d %5d  %10.4f  %7.4f  %9d\n",
               pairs[i].a, pairs[i].b,
               pairs[i].h_joint, pairs[i].improvement,
               pairs[i].n_patterns);

    /* === 4. Enumerate the actual top skip-2-gram patterns === */
    printf("\n=== Top skip-2-gram patterns (by count) ===\n");
    printf("Patterns from the best offset pair\n\n");

    if (n_pairs > 0) {
        int best_a = pairs[0].a;
        int best_b = pairs[0].b;
        printf("Best offset pair: (%d, %d)\n\n", best_a, best_b);

        /* Collect all patterns for this offset pair */
        #define MAX_SKIP2 4096
        Skip2Gram patterns[MAX_SKIP2];
        int np = 0;

        /* Count (xa, xb, y) triples */
        int min_start = best_b;
        for (int t = min_start; t < len - 1; t++) {
            int xa = data[t - best_a + 1];
            int xb = data[t - best_b + 1];
            int y = data[t + 1];

            /* Find or insert */
            int found = -1;
            for (int i = 0; i < np; i++) {
                if (patterns[i].x_a == xa && patterns[i].x_b == xb &&
                    patterns[i].y == y) {
                    found = i;
                    break;
                }
            }
            if (found >= 0) {
                patterns[found].count++;
            } else if (np < MAX_SKIP2) {
                patterns[np].x_a = xa;
                patterns[np].off_a = best_a;
                patterns[np].x_b = xb;
                patterns[np].off_b = best_b;
                patterns[np].y = y;
                patterns[np].count = 1;
                patterns[np].bpc_gain = 0;
                np++;
            }
        }

        /* Sort by count */
        for (int i = 0; i < np - 1; i++)
            for (int j = i + 1; j < np; j++)
                if (patterns[j].count > patterns[i].count) {
                    Skip2Gram tmp = patterns[i];
                    patterns[i] = patterns[j];
                    patterns[j] = tmp;
                }

        printf("x_a  @off_a  x_b  @off_b  ->  y     count\n");
        char safe(int c) { return (c >= 32 && c < 127) ? c : '.'; }
        for (int i = 0; i < 30 && i < np; i++) {
            Skip2Gram* p = &patterns[i];
            printf("'%c'  @%-5d  '%c'  @%-5d  ->  '%c'   %5d\n",
                   safe(p->x_a), p->off_a,
                   safe(p->x_b), p->off_b,
                   safe(p->y), p->count);
        }
        printf("\nTotal distinct skip-2-gram patterns: %d\n", np);
    }

    /* === 5. Full bpc evaluation with skip-2-gram backoff === */
    printf("\n=== Skip-2-gram backoff predictor ===\n");
    printf("For each position, use (x@1, x@best_b) if seen, else x@1, else marginal\n\n");

    if (n_pairs > 0) {
        int best_b = pairs[0].b;
        double total_bits = 0;
        int n_skip2_used = 0;
        int n_bigram_used = 0;
        int n_marginal_used = 0;

        for (int t = 0; t < len - 1; t++) {
            int y = data[t + 1];

            if (t >= best_b) {
                int xa = data[t];          /* offset 1 */
                int xb = data[t - best_b + 1]; /* offset best_b */

                /* Look up P(y | xa@1, xb@best_b) from training data */
                /* Count all outputs for this (xa, xb) pair */
                int yc[N_BYTES] = {0};
                int total = 0;
                for (int s = best_b; s < len - 1; s++) {
                    if (data[s] == xa && data[s - best_b + 1] == xb) {
                        yc[data[s + 1]]++;
                        total++;
                    }
                }

                if (total > 1 && yc[y] > 0) {
                    /* Use skip-2-gram prediction */
                    double p = (double)yc[y] / total;
                    total_bits -= log2(p);
                    n_skip2_used++;
                    continue;
                }
            }

            /* Backoff to bigram (offset 1) */
            if (t >= 1) {
                int xa = data[t];
                int yc[N_BYTES] = {0};
                int total = 0;
                for (int s = 1; s < len - 1; s++) {
                    if (data[s] == xa) {
                        yc[data[s + 1]]++;
                        total++;
                    }
                }
                if (total > 0 && yc[y] > 0) {
                    double p = (double)yc[y] / total;
                    total_bits -= log2(p);
                    n_bigram_used++;
                    continue;
                }
            }

            /* Backoff to marginal */
            if (out_count[y] > 0) {
                double p = (double)out_count[y] / n_outputs;
                total_bits -= log2(p);
            } else {
                total_bits += 8.0;
            }
            n_marginal_used++;
        }

        double bpc = total_bits / n_outputs;
        printf("Skip-2-gram used: %d positions (%.1f%%)\n",
               n_skip2_used, 100.0 * n_skip2_used / n_outputs);
        printf("Bigram fallback:  %d positions (%.1f%%)\n",
               n_bigram_used, 100.0 * n_bigram_used / n_outputs);
        printf("Marginal fallback: %d positions (%.1f%%)\n",
               n_marginal_used, 100.0 * n_marginal_used / n_outputs);
        printf("Total bpc: %.4f\n", bpc);
    }

    /* === 6. Exhaustive: try ALL offset pairs for skip-2-gram backoff === */
    printf("\n=== Exhaustive skip-2-gram search ===\n");
    printf("Try each second offset (with offset 1 fixed) as skip-2-gram predictor\n\n");

    printf("off_b  bpc    skip2%%  bigram%%  marginal%%\n");
    double best_bpc = 99;
    int best_off = 0;

    for (int ob = 2; ob <= max_off && ob < len; ob++) {
        double total_bits = 0;
        int n_s2 = 0, n_bi = 0, n_mg = 0;

        for (int t = 0; t < len - 1; t++) {
            int y = data[t + 1];

            if (t >= ob) {
                int xa = data[t];
                int xb = data[t - ob + 1];

                int yc[N_BYTES] = {0};
                int total = 0;
                for (int s = ob; s < len - 1; s++) {
                    if (data[s] == xa && data[s - ob + 1] == xb) {
                        yc[data[s + 1]]++;
                        total++;
                    }
                }
                if (total > 1 && yc[y] > 0) {
                    double p = (double)yc[y] / total;
                    total_bits -= log2(p);
                    n_s2++;
                    continue;
                }
            }

            if (t >= 1) {
                int xa = data[t];
                int yc[N_BYTES] = {0};
                int total = 0;
                for (int s = 1; s < len - 1; s++) {
                    if (data[s] == xa) {
                        yc[data[s + 1]]++;
                        total++;
                    }
                }
                if (total > 0 && yc[y] > 0) {
                    double p = (double)yc[y] / total;
                    total_bits -= log2(p);
                    n_bi++;
                    continue;
                }
            }

            if (out_count[y] > 0) {
                double p = (double)out_count[y] / n_outputs;
                total_bits -= log2(p);
            } else {
                total_bits += 8.0;
            }
            n_mg++;
        }

        double bpc = total_bits / n_outputs;
        printf("%5d  %.3f  %5.1f%%  %6.1f%%  %8.1f%%\n",
               ob, bpc,
               100.0 * n_s2 / n_outputs,
               100.0 * n_bi / n_outputs,
               100.0 * n_mg / n_outputs);

        if (bpc < best_bpc) {
            best_bpc = bpc;
            best_off = ob;
        }
    }

    printf("\nBest second offset: %d (bpc=%.4f)\n", best_off, best_bpc);
    printf("Compare: bigram-only bpc = %.4f\n", h_y_given_x_best[1]);

    free(pairs);
    return 0;
}
