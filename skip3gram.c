/*
 * skip3gram.c - Skip-3-gram analysis for the pattern-chain UM
 *
 * Extends skip-2-gram: uses THREE input bytes at offsets (a, b, c)
 * to predict the output byte. Measures H(Y | X@a, X@b, X@c).
 *
 * Key question: how much does a third non-contiguous byte improve
 * over the best skip-2-gram (offsets 1,2 → 0.817 bpc)?
 *
 * Usage: ./skip3gram <datafile>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_DATA 2048
#define MAX_OFFSET 20  /* keep combinatorics manageable */
#define NBYTES 256

static unsigned char data[MAX_DATA];
static int data_len;

/* H(Y) - marginal entropy of output */
static double marginal_bpc(void) {
    int count[NBYTES] = {0};
    int N = data_len - 1;
    for (int i = 1; i < data_len; i++)
        count[data[i]]++;
    double H = 0;
    for (int b = 0; b < NBYTES; b++) {
        if (count[b] == 0) continue;
        double p = (double)count[b] / N;
        H -= p * log2(p);
    }
    return H;
}

/* H(Y | X@a) - conditional entropy with single offset */
static double single_offset_bpc(int off_a) {
    /* count[xa][y] */
    int count[NBYTES][NBYTES];
    int ctx_count[NBYTES];
    memset(count, 0, sizeof(count));
    memset(ctx_count, 0, sizeof(ctx_count));

    int N = 0;
    for (int i = off_a; i < data_len - 1; i++) {
        int xa = data[i - off_a];
        int y  = data[i + 1];
        /* predict data[i+1] given data[i-off_a+1]... actually:
           output at position i+1, input at position i+1 - off_a */
        /* Let's be consistent: output = data[t], input at offset d = data[t-d] */
    }
    /* Redo: output at position t, input at t-off_a */
    memset(count, 0, sizeof(count));
    memset(ctx_count, 0, sizeof(ctx_count));
    N = 0;
    for (int t = off_a; t < data_len; t++) {
        int xa = data[t - off_a];
        int y  = data[t];
        count[xa][y]++;
        ctx_count[xa]++;
        N++;
    }

    double H = 0;
    for (int xa = 0; xa < NBYTES; xa++) {
        if (ctx_count[xa] == 0) continue;
        for (int y = 0; y < NBYTES; y++) {
            if (count[xa][y] == 0) continue;
            double p = (double)count[xa][y] / ctx_count[xa];
            H -= (double)count[xa][y] / N * log2(p);
        }
    }
    return H;
}

/*
 * H(Y | X@a, X@b) - conditional entropy with two offsets
 * Returns the bpc and number of patterns via pointers
 */
static double skip2_bpc(int off_a, int off_b, int *n_patterns) {
    /* Use hash table: key = (xa, xb), value = count[y] array */
    /* For simplicity, use xa*256+xb as key, with a flat array */
    /* 256*256 = 65536 contexts, each with 256 output bins */
    /* That's 16M ints = 64MB. Too much. Use sparse approach. */

    /* Sparse: linked list of (xa, xb) -> count[256] */
    typedef struct Entry {
        int xa, xb;
        int count[NBYTES];
        int total;
        struct Entry *next;
    } Entry;

    #define HASH_SIZE 65537
    Entry *table[HASH_SIZE];
    memset(table, 0, sizeof(table));

    int max_off = off_a > off_b ? off_a : off_b;
    int N = 0;

    for (int t = max_off; t < data_len; t++) {
        int xa = data[t - off_a];
        int xb = data[t - off_b];
        int y  = data[t];

        unsigned int h = (xa * 257 + xb) % HASH_SIZE;
        Entry *e = table[h];
        while (e && (e->xa != xa || e->xb != xb)) e = e->next;
        if (!e) {
            e = calloc(1, sizeof(Entry));
            e->xa = xa; e->xb = xb;
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

    /* Free */
    for (int i = 0; i < HASH_SIZE; i++) {
        Entry *e = table[i];
        while (e) { Entry *next = e->next; free(e); e = next; }
    }

    return H;
}

/*
 * H(Y | X@a, X@b, X@c) - conditional entropy with three offsets
 */
static double skip3_bpc(int off_a, int off_b, int off_c, int *n_patterns) {
    typedef struct Entry {
        int xa, xb, xc;
        int count[NBYTES];
        int total;
        struct Entry *next;
    } Entry;

    #define HASH3_SIZE 131071
    Entry *table[HASH3_SIZE];
    memset(table, 0, sizeof(table));

    int max_off = off_a;
    if (off_b > max_off) max_off = off_b;
    if (off_c > max_off) max_off = off_c;
    int N = 0;

    for (int t = max_off; t < data_len; t++) {
        int xa = data[t - off_a];
        int xb = data[t - off_b];
        int xc = data[t - off_c];
        int y  = data[t];

        unsigned int h = ((unsigned)(xa * 65537 + xb * 257 + xc)) % HASH3_SIZE;
        Entry *e = table[h];
        while (e && (e->xa != xa || e->xb != xb || e->xc != xc)) e = e->next;
        if (!e) {
            e = calloc(1, sizeof(Entry));
            e->xa = xa; e->xb = xb; e->xc = xc;
            e->next = table[h];
            table[h] = e;
        }
        e->count[y]++;
        e->total++;
        N++;
    }

    double H = 0;
    int patterns = 0;
    for (int i = 0; i < HASH3_SIZE; i++) {
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

    /* Free */
    for (int i = 0; i < HASH3_SIZE; i++) {
        Entry *e = table[i];
        while (e) { Entry *next = e->next; free(e); e = next; }
    }

    return H;
}

/*
 * Backoff predictor: use (x@1, x@2, x@c) if seen, else (x@1, x@2),
 * else x@1, else marginal.
 */
static void skip3_backoff(int off_c) {
    /* Build tables for each level */
    /* Level 3: (x@1, x@2, x@c) -> count[y] */
    typedef struct E3 {
        int xa, xb, xc;
        int count[NBYTES];
        int total;
        struct E3 *next;
    } E3;

    typedef struct E2 {
        int xa, xb;
        int count[NBYTES];
        int total;
        struct E2 *next;
    } E2;

    #define H3 131071
    #define H2 65537
    E3 *t3[H3]; memset(t3, 0, sizeof(t3));
    E2 *t2[H2]; memset(t2, 0, sizeof(t2));
    int uni_count[NBYTES][NBYTES]; memset(uni_count, 0, sizeof(uni_count));
    int uni_total[NBYTES]; memset(uni_total, 0, sizeof(uni_total));
    int marginal[NBYTES]; memset(marginal, 0, sizeof(marginal));

    int max_off = 2;
    if (off_c > max_off) max_off = off_c;

    /* Build all tables from training data (= all data, since this is memorization) */
    for (int t = max_off; t < data_len; t++) {
        int x1 = data[t - 1];
        int x2 = data[t - 2];
        int xc = data[t - off_c];
        int y  = data[t];

        /* Level 3 */
        unsigned int h = ((unsigned)(x1 * 65537 + x2 * 257 + xc)) % H3;
        E3 *e3 = t3[h];
        while (e3 && (e3->xa != x1 || e3->xb != x2 || e3->xc != xc)) e3 = e3->next;
        if (!e3) {
            e3 = calloc(1, sizeof(E3));
            e3->xa = x1; e3->xb = x2; e3->xc = xc;
            e3->next = t3[h];
            t3[h] = e3;
        }
        e3->count[y]++;
        e3->total++;

        /* Level 2 */
        h = (x1 * 257 + x2) % H2;
        E2 *e2 = t2[h];
        while (e2 && (e2->xa != x1 || e2->xb != x2)) e2 = e2->next;
        if (!e2) {
            e2 = calloc(1, sizeof(E2));
            e2->xa = x1; e2->xb = x2;
            e2->next = t2[h];
            t2[h] = e2;
        }
        e2->count[y]++;
        e2->total++;

        /* Level 1 */
        uni_count[x1][y]++;
        uni_total[x1]++;

        marginal[y]++;
    }

    int total = 0;
    for (int b = 0; b < NBYTES; b++) total += marginal[b];

    /* Now predict each position */
    double total_log = 0;
    int n_skip3 = 0, n_skip2 = 0, n_uni = 0, n_marg = 0;
    int N = 0;

    for (int t = max_off; t < data_len; t++) {
        int x1 = data[t - 1];
        int x2 = data[t - 2];
        int xc = data[t - off_c];
        int y  = data[t];
        double p = 0;

        /* Try level 3 */
        unsigned int h = ((unsigned)(x1 * 65537 + x2 * 257 + xc)) % H3;
        E3 *e3 = t3[h];
        while (e3 && (e3->xa != x1 || e3->xb != x2 || e3->xc != xc)) e3 = e3->next;
        if (e3 && e3->total > 0) {
            p = (double)e3->count[y] / e3->total;
            if (p > 0) { n_skip3++; goto done; }
        }

        /* Try level 2 */
        h = (x1 * 257 + x2) % H2;
        E2 *e2 = t2[h];
        while (e2 && (e2->xa != x1 || e2->xb != x2)) e2 = e2->next;
        if (e2 && e2->total > 0) {
            p = (double)e2->count[y] / e2->total;
            if (p > 0) { n_skip2++; goto done; }
        }

        /* Try level 1 */
        if (uni_total[x1] > 0) {
            p = (double)uni_count[x1][y] / uni_total[x1];
            if (p > 0) { n_uni++; goto done; }
        }

        /* Marginal */
        p = (double)marginal[y] / total;
        n_marg++;

done:
        if (p > 0) total_log += log2(p);
        else total_log += log2(1.0 / NBYTES); /* uniform fallback */
        N++;
    }

    double bpc = -total_log / N;
    printf("  off_c=%2d  bpc=%.4f  skip3=%d(%.1f%%)  skip2=%d(%.1f%%)  "
           "unigram=%d(%.1f%%)  marginal=%d(%.1f%%)\n",
           off_c, bpc,
           n_skip3, 100.0*n_skip3/N,
           n_skip2, 100.0*n_skip2/N,
           n_uni, 100.0*n_uni/N,
           n_marg, 100.0*n_marg/N);

    /* Free */
    for (int i = 0; i < H3; i++) {
        E3 *e = t3[i]; while (e) { E3 *n = e->next; free(e); e = n; }
    }
    for (int i = 0; i < H2; i++) {
        E2 *e = t2[i]; while (e) { E2 *n = e->next; free(e); e = n; }
    }
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

    printf("Data: %d bytes, max offset: %d\n\n", data_len, MAX_OFFSET);

    double H_y = marginal_bpc();
    printf("H(output) = %.4f bits (marginal bpc = %.4f)\n\n", H_y, H_y);

    /* Baselines */
    printf("=== Baselines ===\n");
    printf("Bigram (offset 1):           bpc = %.4f\n", single_offset_bpc(1));

    int np2;
    double s2_12 = skip2_bpc(1, 2, &np2);
    printf("Skip-2-gram (1,2) = trigram: bpc = %.4f  patterns = %d\n", s2_12, np2);

    double s2_best = 999;
    int s2_best_b = 2;
    for (int b = 2; b <= MAX_OFFSET; b++) {
        double h = skip2_bpc(1, b, NULL);
        if (h < s2_best) { s2_best = h; s2_best_b = b; }
    }
    printf("Best skip-2-gram (1,%d):     bpc = %.4f\n\n", s2_best_b, s2_best);

    /* Skip-3-gram: fix offsets 1,2, vary third */
    printf("=== Skip-3-gram: H(Y | X@1, X@2, X@c) ===\n");
    printf("off_c  H(Y|X@1,X@2,X@c)  improvement  #patterns\n");

    for (int c = 3; c <= MAX_OFFSET; c++) {
        int np;
        double h = skip3_bpc(1, 2, c, &np);
        printf("  %2d       %.4f         %.4f       %d\n",
               c, h, s2_12 - h, np);
    }

    /* Skip-3-gram: fix offset 1, vary second and third */
    printf("\n=== Skip-3-gram: H(Y | X@1, X@b, X@c) for selected pairs ===\n");
    printf("off_b  off_c  H(Y|X@1,X@b,X@c)  #patterns\n");

    int pairs[][2] = {
        {2,3}, {2,4}, {2,5}, {2,8}, {2,10}, {2,15},
        {3,5}, {3,8}, {3,10}, {3,15},
        {4,8}, {4,12}, {5,10}, {5,15},
        {8,15}, {10,15}, {10,20},
        {-1,-1}
    };
    for (int i = 0; pairs[i][0] >= 0; i++) {
        int b = pairs[i][0], c = pairs[i][1];
        int np;
        double h = skip3_bpc(1, b, c, &np);
        printf("  %2d     %2d       %.4f          %d\n", b, c, h, np);
    }

    /* Skip-3-gram backoff predictor */
    printf("\n=== Skip-3-gram backoff predictor (X@1, X@2, X@c) ===\n");
    printf("Uses (x@1, x@2, x@c) if seen, else (x@1, x@2), else x@1, else marginal\n\n");

    for (int c = 3; c <= MAX_OFFSET; c++) {
        skip3_backoff(c);
    }

    /* Best overall: try top skip-3-gram combinations */
    printf("\n=== Top skip-3-gram offset triples by H(Y|Xa,Xb,Xc) ===\n");
    printf("off_a  off_b  off_c  H(Y|...)  #patterns\n");

    typedef struct { int a, b, c; double h; int np; } Result;
    Result results[2000];
    int nresults = 0;

    for (int a = 1; a <= 5; a++) {
        for (int b = a+1; b <= 10; b++) {
            for (int c = b+1; c <= MAX_OFFSET; c++) {
                int np;
                double h = skip3_bpc(a, b, c, &np);
                results[nresults++] = (Result){a, b, c, h, np};
            }
        }
    }

    /* Sort by H ascending */
    for (int i = 0; i < nresults-1; i++)
        for (int j = i+1; j < nresults; j++)
            if (results[j].h < results[i].h) {
                Result tmp = results[i]; results[i] = results[j]; results[j] = tmp;
            }

    int show = nresults < 30 ? nresults : 30;
    for (int i = 0; i < show; i++) {
        printf("  %2d     %2d     %2d    %.4f     %d\n",
               results[i].a, results[i].b, results[i].c,
               results[i].h, results[i].np);
    }

    return 0;
}
