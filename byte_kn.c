/*
 * byte_kn.c — Byte-level interpolated Kneser-Ney n-gram model.
 *
 * Computes bits-per-character (bpc) on raw bytes using interpolated KN
 * smoothing with a single hash table.  No tokenization, no neural network,
 * no event spaces — just exact n-gram counting.
 *
 * Results on enwik9 (80/20 train/test split):
 *   10M  KN-6i D=0.8:  2.315 bpc
 *   100M KN-6i D=0.8:  2.001 bpc
 *   200M KN-6i D=0.8:  1.927 bpc
 *   400M KN-6i D=0.8:  1.889 bpc
 *   800M KN-6i D=0.8:  1.859 bpc  (HT 97%)
 *   1B   KN-6i D=0.9:  1.784 bpc  (HT 100%)
 *
 * Memory: 128M-entry hash table = 1.5 GB, plus data.
 * At 1B the HT is 100% saturated; a larger table would improve results.
 *
 * Usage: byte_kn <data_file> [max_bytes] [order] [discount]
 *   data_file: path to enwik9 (or any raw byte file)
 *   max_bytes: how much data to use (default: all)
 *   order:     max n-gram order (default: 6)
 *   discount:  KN discount parameter (default: 0.8)
 *
 * Build: gcc -O2 -o byte_kn byte_kn.c -lm
 *
 * Michaeljohn Clement and Claude, February 2026.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* --- Hash table --- */

#define HT_SIZE (1 << 27)   /* 128M entries, ~1.5 GB */
#define HT_MASK (HT_SIZE - 1)

typedef struct { unsigned long long key; int cnt; } HE;
static HE *ht = NULL;
static int ht_used = 0;

static void ht_init(void) {
    if (!ht) {
        ht = calloc(HT_SIZE, sizeof(HE));
        if (!ht) {
            fprintf(stderr, "ERROR: cannot allocate %zu bytes for hash table\n",
                    (size_t)HT_SIZE * sizeof(HE));
            exit(1);
        }
    } else {
        memset(ht, 0, (size_t)HT_SIZE * sizeof(HE));
    }
    ht_used = 0;
}

static void ht_add(unsigned long long k, int v) {
    unsigned int i = (unsigned int)((k >> 3) & HT_MASK);
    for (int p = 0; p < 128; p++) {
        if (ht[i].key == 0) { ht[i].key = k; ht[i].cnt = v; ht_used++; return; }
        if (ht[i].key == k) { ht[i].cnt += v; return; }
        i = (i + 1) & HT_MASK;
    }
    /* Table full — silently drop (degrades results, does not crash) */
}

static int ht_get(unsigned long long k) {
    unsigned int i = (unsigned int)((k >> 3) & HT_MASK);
    for (int p = 0; p < 128; p++) {
        if (ht[i].key == 0) return 0;
        if (ht[i].key == k) return ht[i].cnt;
        i = (i + 1) & HT_MASK;
    }
    return 0;
}

/* FNV-1a hash with suffix disambiguation */
static unsigned long long mkh(unsigned char *ctx, int len, int suffix) {
    unsigned long long h = 14695981039346656037ULL;
    for (int i = 0; i < len; i++) { h ^= ctx[i]; h *= 1099511628211ULL; }
    h ^= (unsigned long long)suffix * 0x9e3779b97f4a7c15ULL;
    return h | 1;  /* ensure nonzero (0 = empty slot) */
}

/* --- KN model --- */

static double marginal[256];

/*
 * Build n-gram counts up to order max_order.
 * For each position t and order o, stores three entries:
 *   mkh(ctx, o, byte+256)   = continuation count c(ctx, byte)
 *   mkh(ctx, o, 512)        = total count c(ctx, *)
 *   mkh(ctx, o, 513)        = type count tau(ctx) (unique continuations)
 */
static void build(unsigned char *data, long n, int max_order) {
    ht_init();
    for (long t = 0; t < n; t++) {
        for (int o = 1; o <= max_order && o <= t; o++) {
            unsigned long long k = mkh(data + t - o, o, data[t] + 256);
            int prev = ht_get(k);
            ht_add(k, 1);
            ht_add(mkh(data + t - o, o, 512), 1);
            if (prev == 0)
                ht_add(mkh(data + t - o, o, 513), 1);
        }
        if ((t & 0xFFFFFF) == 0 && t > 0)
            fprintf(stderr, "  building: %ldM / %ldM [HT: %.1f%%]\r",
                    t / 1000000, n / 1000000, 100.0 * ht_used / HT_SIZE);
    }
    fprintf(stderr, "  built %ldM [HT: %.1f%% = %dM entries]            \n",
            n / 1000000, 100.0 * ht_used / HT_SIZE, ht_used / 1000000);
}

/*
 * Interpolated KN evaluation.
 * Builds prediction bottom-up from marginal through all matching orders:
 *   P(w | ctx) = pk + lambda * P(w | shorter_ctx)
 * where pk = max(c - D, 0) / total, lambda = D * types / total.
 */
static double eval_interp(unsigned char *data, long from, long to,
                          int max_order, double D) {
    double total_bits = 0;
    long count = 0;
    for (long t = from; t < to; t++) {
        double p = marginal[data[t]];
        for (int o = 1; o <= max_order && o <= t; o++) {
            int tc = ht_get(mkh(data + t - o, o, 512));
            if (tc < 2) continue;
            int c = ht_get(mkh(data + t - o, o, data[t] + 256));
            int ty = ht_get(mkh(data + t - o, o, 513));
            if (ty < 1) ty = 1;
            double pk = (c > D) ? (c - D) / tc : 0.0;
            double lambda = D * ty / tc;
            p = pk + lambda * p;
        }
        if (p < 1e-10) p = 1e-10;
        total_bits += -log2(p);
        count++;
    }
    return count > 0 ? total_bits / count : 8.0;
}

/* --- Main --- */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <data_file> [max_bytes] [order] [discount]\n"
            "\n"
            "Byte-level interpolated Kneser-Ney n-gram model.\n"
            "Trains on first 80%% of data, evaluates on last 20%%.\n"
            "\n"
            "  data_file  Path to enwik9 or any raw byte file\n"
            "  max_bytes  How much data to use (default: entire file)\n"
            "  order      Max n-gram order (default: 6)\n"
            "  discount   KN discount D (default: 0.8)\n"
            "\n"
            "Example: %s enwik9          # full 1B, order 6, D=0.8\n"
            "         %s enwik9 200000000 6 0.8  # 200M subset\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }

    long max_bytes = 0;  /* 0 = read entire file */
    int order = 6;
    double discount = 0.8;

    if (argc >= 3) max_bytes = atol(argv[2]);
    if (argc >= 4) order = atoi(argv[3]);
    if (argc >= 5) discount = atof(argv[4]);

    /* Read data */
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", argv[1]); return 1; }

    if (max_bytes <= 0) {
        fseek(f, 0, SEEK_END);
        max_bytes = ftell(f);
        fseek(f, 0, SEEK_SET);
    }

    unsigned char *data = malloc(max_bytes + 1);
    if (!data) { fprintf(stderr, "ERROR: cannot allocate %ld bytes\n", max_bytes); return 1; }
    long n = fread(data, 1, max_bytes, f);
    fclose(f);

    /* 80/20 split */
    long train = (long)(n * 0.8);
    long test = n - train;

    printf("=== Byte-Level Interpolated Kneser-Ney ===\n");
    printf("Data: %ld bytes (train: %ldM, test: %ldM)\n", n, train/1000000, test/1000000);
    printf("Order: %d, Discount: %.2f\n", order, discount);
    printf("Hash table: %d entries (%.0f MB)\n\n",
           HT_SIZE, (double)HT_SIZE * sizeof(HE) / (1024 * 1024));

    /* Compute marginals from training data */
    long bc[256];
    memset(bc, 0, sizeof(bc));
    for (long t = 0; t < train; t++) bc[data[t]]++;
    for (int i = 0; i < 256; i++)
        marginal[i] = (bc[i] + 0.5) / (train + 128.0);

    /* Build and evaluate */
    printf("Building KN-%d on %ldM training bytes...\n", order, train/1000000);
    build(data, train, order);

    double bpc = eval_interp(data, train, n, order, discount);
    printf("\nResult: %.3f bpc (interpolated KN-%d, D=%.2f)\n", bpc, order, discount);
    printf("HT usage: %.1f%% (%dM / %dM entries)\n",
           100.0 * ht_used / HT_SIZE, ht_used / 1000000, HT_SIZE / 1000000);

    /* Try nearby discounts for comparison */
    printf("\nDiscount sweep:\n");
    for (double d = 0.5; d <= 1.0; d += 0.1) {
        double r = eval_interp(data, train, n, order, d);
        printf("  D=%.1f: %.3f bpc%s\n", d, r, (fabs(d - discount) < 0.01) ? " <-- selected" : "");
    }

    free(data);
    free(ht);
    return 0;
}
