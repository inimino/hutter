/*
 * skip2_survival.c — Which skip-2-gram patterns survive DSS doubling?
 *
 * Compares patterns found in bytes [0,1024) with those in [0,2048).
 * Patterns that recur in the second half are structural (XML, words).
 * Patterns that don't are artifacts of the specific 1024-byte dataset.
 *
 * This is the key test: with DSS=1024 the RNN can't generalize beyond
 * these data-terms. When we double, we can predict which W_hh changes
 * the RNN needs: strengthen surviving patterns, weaken artifacts.
 *
 * Usage: skip2_survival <data_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N_BYTES 256
#define MAX_PATTERNS 8192

typedef struct {
    unsigned char xa, xb, y;
    int off_a, off_b;
    int count_first;    /* count in [0, 1024) */
    int count_second;   /* count in [1024, 2048) */
    int count_total;    /* count in [0, 2048) */
    double bpc_first;   /* -log2(p) in first half */
    double bpc_second;  /* -log2(p) in second half (0 if absent) */
} Pattern;

static char safe(int c) { return (c >= 32 && c < 127) ? c : '.'; }

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_file>\n", argv[0]);
        return 1;
    }

    FILE* df = fopen(argv[1], "rb");
    if (!df) { perror("data"); return 1; }
    unsigned char data[2048];
    int len = fread(data, 1, 2048, df);
    fclose(df);

    if (len < 2048) {
        fprintf(stderr, "Need at least 2048 bytes, got %d\n", len);
        return 1;
    }
    printf("Data: %d bytes (comparing first 1024 vs full 2048)\n\n", len);

    /* We'll analyze offset pair (1, 2) — the trigram-like skip-2-gram
     * since it's the best predictor. Also do (1, 4) and (1, 8) for
     * skip patterns that require W_hh memory. */

    int offsets[][2] = {{1, 2}, {1, 4}, {1, 8}, {2, 6}, {3, 12}};
    int n_offset_pairs = 5;

    for (int op = 0; op < n_offset_pairs; op++) {
        int oa = offsets[op][0];
        int ob = offsets[op][1];

        printf("========================================\n");
        printf("Offset pair (%d, %d)\n", oa, ob);
        printf("========================================\n\n");

        Pattern pats[MAX_PATTERNS];
        int np = 0;

        /* Count patterns in first half [0, 1024) */
        for (int t = ob; t < 1023; t++) {
            int xa = data[t - oa + 1];
            int xb = data[t - ob + 1];
            int y = data[t + 1];

            int found = -1;
            for (int i = 0; i < np; i++) {
                if (pats[i].xa == xa && pats[i].xb == xb && pats[i].y == y) {
                    found = i; break;
                }
            }
            if (found >= 0) {
                pats[found].count_first++;
            } else if (np < MAX_PATTERNS) {
                pats[np].xa = xa;
                pats[np].xb = xb;
                pats[np].y = y;
                pats[np].off_a = oa;
                pats[np].off_b = ob;
                pats[np].count_first = 1;
                pats[np].count_second = 0;
                pats[np].count_total = 0;
                np++;
            }
        }

        /* Count patterns in second half [1024, 2048) */
        /* Note: context bytes can come from first half */
        for (int t = 1024; t < 2047; t++) {
            if (t - ob + 1 < 0) continue;
            int xa = data[t - oa + 1];
            int xb = data[t - ob + 1];
            int y = data[t + 1];

            int found = -1;
            for (int i = 0; i < np; i++) {
                if (pats[i].xa == xa && pats[i].xb == xb && pats[i].y == y) {
                    found = i; break;
                }
            }
            if (found >= 0) {
                pats[found].count_second++;
            } else if (np < MAX_PATTERNS) {
                /* New pattern only in second half */
                pats[np].xa = xa;
                pats[np].xb = xb;
                pats[np].y = y;
                pats[np].off_a = oa;
                pats[np].off_b = ob;
                pats[np].count_first = 0;
                pats[np].count_second = 1;
                pats[np].count_total = 0;
                np++;
            }
        }

        /* Compute totals */
        for (int i = 0; i < np; i++)
            pats[i].count_total = pats[i].count_first + pats[i].count_second;

        /* Classify patterns */
        int n_survive = 0, n_artifact_first = 0, n_new_second = 0, n_both = 0;
        int survive_count = 0, artifact_count = 0, new_count = 0;

        for (int i = 0; i < np; i++) {
            if (pats[i].count_first > 0 && pats[i].count_second > 0) {
                n_survive++;
                survive_count += pats[i].count_total;
            } else if (pats[i].count_first > 0 && pats[i].count_second == 0) {
                n_artifact_first++;
                artifact_count += pats[i].count_first;
            } else {
                n_new_second++;
                new_count += pats[i].count_second;
            }
        }

        printf("Pattern survival:\n");
        printf("  Surviving (both halves):  %4d patterns, %5d occurrences\n",
               n_survive, survive_count);
        printf("  Artifacts (first only):   %4d patterns, %5d occurrences\n",
               n_artifact_first, artifact_count);
        printf("  New (second only):        %4d patterns, %5d occurrences\n",
               n_new_second, new_count);
        printf("  Survival rate: %.1f%% of first-half patterns\n",
               100.0 * n_survive / (n_survive + n_artifact_first));
        printf("  Survival rate by count: %.1f%% of first-half occurrences\n\n",
               100.0 * survive_count /
               (survive_count + artifact_count + 0.001));

        /* Sort surviving patterns by count_first (strongest predictions) */
        for (int i = 0; i < np - 1; i++)
            for (int j = i + 1; j < np; j++)
                if (pats[j].count_first > pats[i].count_first) {
                    Pattern tmp = pats[i]; pats[i] = pats[j]; pats[j] = tmp;
                }

        /* Show top surviving patterns */
        printf("Top 20 SURVIVING patterns (structural):\n");
        printf("x_a  x_b  ->  y     1st   2nd   ratio\n");
        int shown = 0;
        for (int i = 0; i < np && shown < 20; i++) {
            if (pats[i].count_first == 0 || pats[i].count_second == 0) continue;
            double ratio = (double)pats[i].count_second / pats[i].count_first;
            printf("'%c'  '%c'  ->  '%c'   %4d  %4d  %.2f\n",
                   safe(pats[i].xa), safe(pats[i].xb), safe(pats[i].y),
                   pats[i].count_first, pats[i].count_second, ratio);
            shown++;
        }

        /* Show top artifact patterns (first half only, by count) */
        printf("\nTop 15 ARTIFACT patterns (first half only — won't generalize):\n");
        printf("x_a  x_b  ->  y     count\n");
        shown = 0;
        for (int i = 0; i < np && shown < 15; i++) {
            if (pats[i].count_second != 0 || pats[i].count_first == 0) continue;
            printf("'%c'  '%c'  ->  '%c'   %4d\n",
                   safe(pats[i].xa), safe(pats[i].xb), safe(pats[i].y),
                   pats[i].count_first);
            shown++;
        }

        /* Show top new patterns (second half only) */
        printf("\nTop 15 NEW patterns (second half only — RNN must learn):\n");
        printf("x_a  x_b  ->  y     count\n");
        /* Re-sort by count_second for new patterns */
        for (int i = 0; i < np - 1; i++)
            for (int j = i + 1; j < np; j++)
                if (pats[j].count_second > pats[i].count_second &&
                    pats[j].count_first == 0 && pats[i].count_first == 0) {
                    Pattern tmp = pats[i]; pats[i] = pats[j]; pats[j] = tmp;
                }

        shown = 0;
        for (int i = 0; i < np && shown < 15; i++) {
            if (pats[i].count_first != 0 || pats[i].count_second == 0) continue;
            printf("'%c'  '%c'  ->  '%c'   %4d\n",
                   safe(pats[i].xa), safe(pats[i].xb), safe(pats[i].y),
                   pats[i].count_second);
            shown++;
        }

        /* Stability metric: correlation between first and second half counts */
        /* Only for patterns that appear in both halves */
        double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
        int ns = 0;
        for (int i = 0; i < np; i++) {
            if (pats[i].count_first > 0 && pats[i].count_second > 0) {
                double x = pats[i].count_first;
                double y = pats[i].count_second;
                sx += x; sy += y;
                sxx += x*x; syy += y*y; sxy += x*y;
                ns++;
            }
        }
        if (ns > 1) {
            double mx = sx/ns, my = sy/ns;
            double r = (sxy/ns - mx*my) /
                       (sqrt(sxx/ns - mx*mx) * sqrt(syy/ns - my*my) + 1e-10);
            printf("\nCount correlation (surviving patterns): r=%.3f (n=%d)\n", r, ns);
        }

        printf("\n");
    }

    /* === Prediction: what happens when RNN trains on 2048 === */
    printf("========================================\n");
    printf("PREDICTION: RNN behavior on DSS doubling\n");
    printf("========================================\n\n");

    printf("When the RNN trains on 2048 bytes instead of 1024:\n\n");
    printf("1. SURVIVING patterns: W_hh entries that encode these\n");
    printf("   should STRENGTHEN. These are real structural patterns.\n\n");
    printf("2. ARTIFACT patterns: W_hh entries that encode these\n");
    printf("   should WEAKEN or repurpose. The RNN wasted capacity on\n");
    printf("   patterns that were accidents of the specific 1024 bytes.\n\n");
    printf("3. NEW patterns: W_hh must learn new entries. These are\n");
    printf("   structural patterns that didn't appear in the first 1024\n");
    printf("   bytes but do appear in the second 1024.\n\n");
    printf("4. The ratio of surviving:artifact patterns bounds the\n");
    printf("   useful capacity of the RNN at this DSS.\n");

    return 0;
}
