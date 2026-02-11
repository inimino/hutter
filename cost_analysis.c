/*
 * cost_analysis.c — Measure actual wall-clock cost of analytic construction
 * vs SGD training for the 128-hidden RNN.
 *
 * Counts FLOPs precisely for each approach and validates with timing.
 * Produces numbers for the computational cost paper.
 *
 * Usage: cost_analysis <data_file> [max_bytes]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Model dimensions */
#define H 128
#define V 256
#define G 16    /* offsets */
#define BPTT 50

/* FLOP counting */
typedef struct {
    long long madd;     /* multiply-accumulate (= 2 FLOPs each) */
    long long add;      /* additions */
    long long mul;      /* multiplies */
    long long transcend; /* exp, log, tanh, etc. */
    long long intop;    /* integer operations (counting) */
} FlopCount;

void print_flops(const char *label, FlopCount *f, double wall_sec) {
    long long total_flop = f->madd * 2 + f->add + f->mul + f->transcend;
    long long total_with_int = total_flop + f->intop; /* rough: 1 intop ~ 0.5 flop */
    printf("  %-30s  MADDs: %12lld  FLOPs: %12lld  IntOps: %12lld  Total: %12lld",
           label, f->madd, total_flop, f->intop, total_with_int);
    if (wall_sec > 0) printf("  Wall: %.4fs", wall_sec);
    printf("\n");
}

/* ========== SGD TRAINING FLOP COUNT ========== */

/* Per-timestep forward pass */
void count_forward_step(FlopCount *f) {
    /* h = tanh(W_x[byte] + W_h @ h_prev + b_h) */
    /* W_x[byte]: lookup, 0 MADDs (one-hot) but H copies */
    f->add += H;  /* add W_x row to accumulator */
    /* W_h @ h_prev: H x H matrix-vector */
    f->madd += (long long)H * H;
    /* + b_h */
    f->add += H;
    /* tanh(.) */
    f->transcend += H;

    /* y = softmax(W_y @ h + b_y) */
    /* W_y @ h: V x H matrix-vector */
    f->madd += (long long)V * H;
    /* + b_y */
    f->add += V;
    /* softmax: V exp + V add + V div */
    f->transcend += V;  /* exp */
    f->add += V;        /* sum */
    f->mul += V;        /* divide (count as mul) */
}

/* Per-timestep backward pass */
void count_backward_step(FlopCount *f) {
    /* delta_y = y - one_hot: V subtractions */
    f->add += V;

    /* grad_W_y += delta_y outer h: V x H */
    f->madd += (long long)V * H;

    /* delta_h = W_y^T @ delta_y: H x V (transpose multiply) */
    f->madd += (long long)H * V;

    /* delta_h *= (1 - h^2): elementwise tanh derivative */
    f->mul += H;  /* h^2 */
    f->add += H;  /* 1 - h^2 */
    f->mul += H;  /* multiply by delta */

    /* grad_W_h += delta_h outer h_prev: H x H */
    f->madd += (long long)H * H;

    /* delta_h_prev = W_h^T @ delta_h: H x H */
    f->madd += (long long)H * H;

    /* grad_W_x[byte] += delta_h: H additions */
    f->add += H;
}

/* Count SGD training FLOPs for N bytes, E epochs */
void count_sgd(int N, int E, FlopCount *f) {
    memset(f, 0, sizeof(*f));
    long long segments = (long long)N / BPTT;
    long long steps_per_epoch = segments;

    /* Per segment: BPTT forward steps + BPTT backward steps */
    FlopCount fwd = {0}, bwd = {0};
    count_forward_step(&fwd);
    count_backward_step(&bwd);

    FlopCount per_step = {0};
    per_step.madd = (fwd.madd + bwd.madd) * BPTT;
    per_step.add = (fwd.add + bwd.add) * BPTT;
    per_step.mul = (fwd.mul + bwd.mul) * BPTT;
    per_step.transcend = (fwd.transcend + bwd.transcend) * BPTT;

    f->madd = per_step.madd * steps_per_epoch * E;
    f->add = per_step.add * steps_per_epoch * E;
    f->mul = per_step.mul * steps_per_epoch * E;
    f->transcend = per_step.transcend * steps_per_epoch * E;
}

/* ========== ANALYTIC CONSTRUCTION FLOP COUNT ========== */

/* Step 1: Count skip-bigrams */
void count_skipbigram_counting(int N, FlopCount *f) {
    /* For each position t, for each offset g:
     *   compute index p = t - g - 1  (2 int ops)
     *   bounds check                  (1 int op)
     *   lookup data[p], data[t]       (2 int ops)
     *   compute array index           (2 int ops: g*256*256 + data[p]*256 + data[t])
     *   increment count               (1 int op)
     *   increment row_total           (1 int op)
     * Total: ~9 int ops per (t, g)
     */
    f->intop += (long long)N * G * 9;
}

/* Step 2: Compute marginals */
void count_marginals(int N, FlopCount *f) {
    /* Scan data, count byte frequencies: N int ops */
    f->intop += N;
    /* Compute probabilities: 256 divides */
    f->mul += 256;
    /* Compute log marginals: 256 logs */
    f->transcend += 256;
}

/* Step 3: Compute log-ratio tables (analytic W_y) */
void count_logratio_tables(FlopCount *f) {
    /* For each g, x, o:
     *   P(o|x) = (count[g][x][o] + smooth) / (row_total[g][x] + smooth_total)
     *   log_ratio = log(P(o|x)) - log(P(o))
     * Operations per entry: 1 add + 1 div + 1 log + 1 sub = 4
     */
    long long entries = (long long)G * V * V;
    f->add += entries;      /* smoothing numerator */
    f->mul += entries;      /* division */
    f->transcend += entries; /* log */
    f->add += entries;      /* subtract log marginal */
}

/* Step 4: Hash construction (W_x) */
void count_wx_construction(FlopCount *f) {
    /* For each of G groups, assign 256 byte->pattern mappings
     * Each: hash computation ~10 int ops */
    f->intop += (long long)G * V * 10;
}

/* Step 5: Diagonal W_h */
void count_wh_construction(FlopCount *f) {
    /* H diagonal entries, each a constant assignment */
    f->intop += H;
}

/* Full analytic construction */
void count_analytic(int N, FlopCount *f) {
    memset(f, 0, sizeof(*f));
    count_skipbigram_counting(N, f);
    count_marginals(N, f);
    count_logratio_tables(f);
    count_wx_construction(f);
    count_wh_construction(f);
}

/* ========== OPTIMIZED W_y FLOP COUNT ========== */

/* Forward pass with fixed W_x, W_h to generate hidden states */
void count_forward_fixed(int N, FlopCount *f) {
    /* Per timestep:
     *   h = tanh(W_x[byte] + W_h @ h_prev + b_h)
     *   Same as training forward, minus W_y computation
     */
    long long per_t = 0;
    f->add += (long long)N * H;         /* W_x lookup add */
    f->madd += (long long)N * H * H;    /* W_h @ h */
    f->add += (long long)N * H;         /* bias */
    f->transcend += (long long)N * H;   /* tanh */
}

/* SGD on W_y only: per epoch */
void count_wy_optim_epoch(int N, FlopCount *f) {
    /* Forward through W_y: V x H matvec per timestep */
    f->madd += (long long)N * V * H;
    /* Softmax: V transcendentals per timestep */
    f->transcend += (long long)N * V;
    f->add += (long long)N * V;
    f->mul += (long long)N * V;
    /* Backward through W_y: delta_y outer h = V x H per timestep */
    f->madd += (long long)N * V * H;
    /* delta_y = y - onehot: V per timestep */
    f->add += (long long)N * V;
}

void count_wy_optimized(int N, int wy_epochs, FlopCount *f) {
    memset(f, 0, sizeof(*f));
    /* First: generate all hidden states */
    count_forward_fixed(N, f);
    /* Then: optimize W_y */
    FlopCount wy_ep = {0};
    count_wy_optim_epoch(N, &wy_ep);
    f->madd += wy_ep.madd * wy_epochs;
    f->add += wy_ep.add * wy_epochs;
    f->mul += wy_ep.mul * wy_epochs;
    f->transcend += wy_ep.transcend * wy_epochs;
}

/* ========== ACTUAL TIMING ========== */

static int skip_cnt[G][256][256];
static int skip_row[G][256];
static double skip_w[G][256][256];
static double log_marg[256];

double time_analytic(unsigned char *data, int n) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Step 1: Count skip-bigrams */
    memset(skip_cnt, 0, sizeof(skip_cnt));
    memset(skip_row, 0, sizeof(skip_row));
    for (int t = 1; t < n; t++) {
        int y = data[t];
        for (int g = 0; g < G; g++) {
            int p = t - g - 1;
            if (p < 0) continue;
            skip_cnt[g][data[p]][y]++;
            skip_row[g][data[p]]++;
        }
    }

    /* Step 2: Marginals */
    int bc[256]; memset(bc, 0, sizeof(bc));
    for (int t = 0; t < n; t++) bc[data[t]]++;
    for (int o = 0; o < 256; o++)
        log_marg[o] = log((bc[o] + 0.5) / (n + 128.0));

    /* Step 3: Log-ratio tables */
    for (int g = 0; g < G; g++)
        for (int x = 0; x < 256; x++) {
            int tot = skip_row[g][x];
            for (int o = 0; o < 256; o++) {
                double p = (skip_cnt[g][x][o] + 0.1) / (tot + 25.6);
                skip_w[g][x][o] = log(p) - log_marg[o];
            }
        }

    /* Step 4-5: W_x hash and W_h diagonal are trivial */

    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

/* Simple SGD simulation timing (one epoch, forward+backward) */
static float Wx[V][H], Wh[H][H], Wy[H][V], bh[H], by[V];
static float h_state[H], h_prev[H], y_out[V];

void init_weights(void) {
    srand(42);
    for (int i = 0; i < V; i++)
        for (int j = 0; j < H; j++) Wx[i][j] = 0.01f * ((float)rand()/RAND_MAX - 0.5f);
    for (int i = 0; i < H; i++)
        for (int j = 0; j < H; j++) Wh[i][j] = 0.01f * ((float)rand()/RAND_MAX - 0.5f);
    for (int i = 0; i < H; i++)
        for (int j = 0; j < V; j++) Wy[i][j] = 0.01f * ((float)rand()/RAND_MAX - 0.5f);
    memset(bh, 0, sizeof(bh));
    memset(by, 0, sizeof(by));
}

double time_sgd_epoch(unsigned char *data, int n) {
    struct timespec t0, t1;
    init_weights();
    memset(h_state, 0, sizeof(h_state));

    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* One epoch: forward + backward per timestep (simplified, no actual BPTT storage) */
    for (int t = 1; t < n; t++) {
        int x = data[t-1];
        /* Forward: h = tanh(Wx[x] + Wh@h_prev + bh) */
        memcpy(h_prev, h_state, sizeof(h_prev));
        for (int i = 0; i < H; i++) {
            float sum = Wx[x][i] + bh[i];
            for (int j = 0; j < H; j++) sum += Wh[i][j] * h_prev[j];
            h_state[i] = tanhf(sum);
        }
        /* Forward: y = softmax(Wy^T @ h + by) */
        float maxv = -1e30f;
        for (int o = 0; o < V; o++) {
            float sum = by[o];
            for (int j = 0; j < H; j++) sum += Wy[j][o] * h_state[j];
            y_out[o] = sum;
            if (sum > maxv) maxv = sum;
        }
        float total = 0;
        for (int o = 0; o < V; o++) { y_out[o] = expf(y_out[o] - maxv); total += y_out[o]; }
        for (int o = 0; o < V; o++) y_out[o] /= total;

        /* Backward: simplified (accumulate gradients but don't update for timing) */
        float dy[V], dh[H];
        for (int o = 0; o < V; o++) dy[o] = y_out[o];
        dy[data[t]] -= 1.0f;
        /* dh = Wy @ dy (note: transposed) */
        for (int i = 0; i < H; i++) {
            dh[i] = 0;
            for (int o = 0; o < V; o++) dh[i] += Wy[i][o] * dy[o];
            dh[i] *= (1.0f - h_state[i] * h_state[i]); /* tanh grad */
        }
        /* We skip actual weight updates for timing purity */
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

/* Dollar cost estimates */
void print_cost(const char *label, double wall_sec, double flops) {
    /* A100 GPU: $2.21/hr (AWS p4d), 312 TFLOP/s fp16, 156 TFLOP/s fp32 */
    double a100_sec = flops / 156e12;
    double a100_cost = a100_sec * (2.21 / 3600.0);

    /* CPU core: $0.034/hr (AWS c6i.xlarge = $0.136/hr, 4 cores) */
    double cpu_cost = wall_sec * (0.034 / 3600.0);

    printf("  %-25s  GPU(A100): %.2e sec ($%.6f)  CPU: %.4f sec ($%.2e)\n",
           label, a100_sec, a100_cost, wall_sec, cpu_cost);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <data> [max]\n", argv[0]); return 1; }
    int max = 10000000;
    if (argc >= 3) max = atoi(argv[2]);

    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "Can't open %s\n", argv[1]); return 1; }
    unsigned char *data = malloc(max + 1);
    int n = fread(data, 1, max, f);
    fclose(f);

    printf("============================================================\n");
    printf("  COMPUTATIONAL COST ANALYSIS: Analytic vs SGD Training\n");
    printf("  Model: %d-hidden RNN, %d vocab, %d offsets, BPTT-%d\n", H, V, G, BPTT);
    printf("  Data: %d bytes\n", n);
    printf("  Parameters: W_x=%d + W_h=%d + W_y=%d + biases=%d = %d total\n",
           V*H, H*H, H*V, H+V, V*H + H*H + H*V + H + V);
    printf("============================================================\n\n");

    /* === FLOP Analysis === */
    printf("=== FLOP ANALYSIS ===\n\n");

    /* Per-timestep breakdown */
    printf("Per-timestep FLOPs:\n");
    FlopCount fwd = {0}, bwd = {0};
    count_forward_step(&fwd);
    count_backward_step(&bwd);
    print_flops("Forward (1 step)", &fwd, 0);
    print_flops("Backward (1 step)", &bwd, 0);
    FlopCount both = {fwd.madd+bwd.madd, fwd.add+bwd.add, fwd.mul+bwd.mul, fwd.transcend+bwd.transcend, 0};
    print_flops("Forward+Backward (1 step)", &both, 0);
    long long flops_per_step = both.madd*2 + both.add + both.mul + both.transcend;
    printf("  Total FLOPs per timestep: %lld (%.0fK)\n\n", flops_per_step, flops_per_step/1e3);

    /* SGD at different scales */
    int epochs[] = {1, 10, 20, 50};
    int n_epochs = 4;
    printf("SGD Training (%d bytes, BPTT-%d):\n", n, BPTT);
    for (int ei = 0; ei < n_epochs; ei++) {
        FlopCount sgd = {0};
        count_sgd(n, epochs[ei], &sgd);
        char label[64];
        sprintf(label, "SGD %d epoch(s)", epochs[ei]);
        print_flops(label, &sgd, 0);
        long long total = sgd.madd*2 + sgd.add + sgd.mul + sgd.transcend;
        printf("  → %.3e FLOPs (%.2f TFLOP)\n", (double)total, total/1e12);
    }
    printf("\n");

    /* Analytic construction */
    printf("Analytic Construction (%d bytes, %d offsets):\n", n, G);
    FlopCount ana = {0};
    count_analytic(n, &ana);
    print_flops("Full analytic", &ana, 0);
    long long ana_total = ana.madd*2 + ana.add + ana.mul + ana.transcend + ana.intop;
    printf("  → %.3e total ops (%.4f GFLOP equiv)\n", (double)ana_total, ana_total/1e9);
    printf("\n");

    /* Optimized W_y */
    int wy_epochs_list[] = {1, 5, 10};
    printf("Optimized W_y (forward pass + W_y SGD):\n");
    for (int ei = 0; ei < 3; ei++) {
        FlopCount opt = {0};
        count_wy_optimized(n, wy_epochs_list[ei], &opt);
        char label[64];
        sprintf(label, "W_y optim %d epoch(s)", wy_epochs_list[ei]);
        print_flops(label, &opt, 0);
        long long total = opt.madd*2 + opt.add + opt.mul + opt.transcend;
        printf("  → %.3e FLOPs (%.2f TFLOP)\n", (double)total, total/1e12);
    }
    printf("\n");

    /* === TIMING === */
    printf("=== WALL-CLOCK TIMING ===\n\n");

    /* Time analytic construction */
    double t_ana = time_analytic(data, n);
    printf("Analytic construction: %.4f seconds\n", t_ana);

    /* Time 1 SGD epoch */
    int timing_n = (n > 1000000) ? 1000000 : n;
    double t_sgd = time_sgd_epoch(data, timing_n);
    double t_sgd_full = t_sgd * ((double)n / timing_n);
    printf("SGD 1 epoch (%d bytes, extrapolated to %d): %.4f sec (%.4f sec actual on %d)\n",
           n, n, t_sgd_full, t_sgd, timing_n);
    printf("\n");

    /* === RATIOS === */
    printf("=== ORDER-OF-MAGNITUDE COMPARISON ===\n\n");

    FlopCount sgd20 = {0};
    count_sgd(n, 20, &sgd20);
    long long sgd20_total = sgd20.madd*2 + sgd20.add + sgd20.mul + sgd20.transcend;

    printf("At N=%d bytes:\n", n);
    printf("  SGD (20 epochs):      %.3e FLOPs = %.2f TFLOP\n", (double)sgd20_total, sgd20_total/1e12);
    printf("  Analytic:             %.3e ops   = %.4f GFLOP\n", (double)ana_total, ana_total/1e9);
    double ratio = (double)sgd20_total / ana_total;
    printf("  Ratio: %.0fx (%.1f orders of magnitude)\n", ratio, log10(ratio));
    printf("\n");

    /* Wall-clock ratio */
    double wall_ratio = (t_sgd_full * 20) / t_ana;
    printf("  Wall-clock ratio: %.0fx (%.1f OoM)\n", wall_ratio, log10(wall_ratio));
    printf("\n");

    /* === SCALING PROJECTION === */
    printf("=== SCALING PROJECTIONS ===\n\n");
    printf("%-8s %-8s %15s %15s %12s %8s\n", "H", "Params", "SGD 20ep FLOP", "Analytic ops", "Ratio", "OoM");
    printf("%-8s %-8s %15s %15s %12s %8s\n", "---", "------", "-------------", "------------", "-----", "---");

    int h_sizes[] = {128, 256, 512, 1024, 2048, 4096};
    for (int hi = 0; hi < 6; hi++) {
        int hh = h_sizes[hi];
        long long params = (long long)V*hh + (long long)hh*hh + (long long)hh*V + hh + V;

        /* SGD per timestep: Wx lookup(H) + Wh@h(H²) + bias(H) + tanh(H) + Wy@h(VH) + softmax(V)
         *                 + bwd: delta_y(V) + gradWy(VH) + Wy^T@dy(HV) + tanhgrad(H) + gradWh(H²) + Wh^T@dh(H²) + gradWx(H)
         * MADDs: fwd: H² + VH. bwd: VH + HV + H² + H² = 2VH + 2H²
         * Total MADDs per step: 3H² + 3VH */
        long long madds_per_step = 3LL*hh*hh + 3LL*V*hh;
        long long flops_per_step_h = madds_per_step * 2; /* each MADD = 2 FLOPs, plus overhead ~10% */
        flops_per_step_h = (long long)(flops_per_step_h * 1.1);

        long long sgd_flops = flops_per_step_h * BPTT * ((long long)n / BPTT) * 20;

        /* Analytic: counting = N*G*9 intops, log-ratios = G*V*V*4 FLOPs, Wx hash = G*V*10 */
        long long analytic_ops = (long long)n * G * 9 + (long long)G * V * V * 4 + (long long)G * V * 10;
        /* Plus W_y assembly: G * H * V muls (mapping log-ratios to H-dimensional space) */
        analytic_ops += (long long)G * hh * V;

        double r = (double)sgd_flops / analytic_ops;
        printf("%-8d %-8lld %15.3e %15.3e %12.0f %8.1f\n",
               hh, params, (double)sgd_flops, (double)analytic_ops, r, log10(r));
    }
    printf("\n");

    /* === DOLLAR COSTS === */
    printf("=== DOLLAR COST ESTIMATES ===\n\n");
    printf("Hardware assumptions:\n");
    printf("  A100 GPU: $2.21/hr, 156 TFLOP/s fp32, 312 TFLOP/s fp16\n");
    printf("  CPU core: $0.034/hr (AWS c6i), ~10 GFLOP/s fp32\n\n");

    double a100_rate = 156e12; /* FLOP/s */
    double a100_cost_per_sec = 2.21 / 3600.0;
    double cpu_rate = 10e9; /* FLOP/s, single core practical */
    double cpu_cost_per_sec = 0.034 / 3600.0;

    printf("%-25s %12s %12s %12s %12s\n", "Method", "GPU sec", "GPU $", "CPU sec", "CPU $");
    printf("%-25s %12s %12s %12s %12s\n", "------", "-------", "-----", "-------", "-----");

    /* SGD 20 epochs */
    double sgd_gpu_sec = (double)sgd20_total / a100_rate;
    double sgd_gpu_cost = sgd_gpu_sec * a100_cost_per_sec;
    double sgd_cpu_sec = t_sgd_full * 20;
    double sgd_cpu_cost = sgd_cpu_sec * cpu_cost_per_sec;
    printf("%-25s %12.4f %12.6f %12.2f %12.6f\n", "SGD (20 epochs)", sgd_gpu_sec, sgd_gpu_cost, sgd_cpu_sec, sgd_cpu_cost);

    /* Analytic */
    double ana_gpu_sec = (double)ana_total / a100_rate;
    double ana_gpu_cost = ana_gpu_sec * a100_cost_per_sec;
    double ana_cpu_sec = t_ana;
    double ana_cpu_cost = ana_cpu_sec * cpu_cost_per_sec;
    printf("%-25s %12.6f %12.9f %12.4f %12.9f\n", "Analytic (zero optim)", ana_gpu_sec, ana_gpu_cost, ana_cpu_sec, ana_cpu_cost);

    printf("\n");
    printf("Cost ratio (SGD/Analytic):\n");
    printf("  GPU: $%.6f / $%.9f = %.0fx\n", sgd_gpu_cost, ana_gpu_cost, sgd_gpu_cost/ana_gpu_cost);
    printf("  CPU: $%.6f / $%.9f = %.0fx\n", sgd_cpu_cost, ana_cpu_cost, sgd_cpu_cost/ana_cpu_cost);
    printf("\n");

    /* Projection to frontier */
    printf("=== FRONTIER PROJECTION (hypothetical) ===\n\n");
    printf("If the analytic construction principle scales to transformer-class models:\n\n");
    long long gpt3_params = 175000000000LL;
    double gpt3_flops = 3.14e23;
    double gpt3_cost = 4600000.0;
    printf("GPT-3 training:  175B params, 3.14e23 FLOPs, ~$4.6M\n");
    /* Analytic equivalent: counting = O(N*context_len), assembly = O(params) */
    /* With N=300B tokens, context=2048: counting ~ 300e9 * 2048 * 10 ~ 6e15 ops */
    /* Assembly ~ 175e9 * 10 ~ 2e12 */
    double analytic_frontier = 6e15 + 2e12;
    printf("Analytic equivalent: ~%.1e ops\n", analytic_frontier);
    printf("Ratio: %.0fx (%.1f OoM)\n", gpt3_flops / analytic_frontier, log10(gpt3_flops / analytic_frontier));
    printf("Projected cost: $%.2f (vs $4.6M)\n", analytic_frontier / a100_rate * a100_cost_per_sec);
    printf("\n");
    printf("NOTE: This projection assumes the analytic construction principle\n");
    printf("generalizes to attention-based architectures. This is NOT yet proven.\n");
    printf("The proven result is for 128-hidden RNNs. The projection is illustrative.\n");

    free(data);
    return 0;
}
