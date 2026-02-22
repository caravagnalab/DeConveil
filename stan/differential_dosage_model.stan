// -----------------------------------------------------------------------------
// Gene-by-gene subtype-aware CN-expression model (log link)
//
// Tumor mean:
//   tumor_mu = sf * purity *
//              exp( b0[s]
//                   + dose_log * b_scaling[s]
//                   + dev      * b_deviation[s] )
//
// where:
//   dose_log = log(CN/2)       (0 at CN=2)
//   dev      = (CN-2)/2        (0 at CN=2)
//
// Full mean:
//   mu = tumor_mu + sf * (1 - purity) * exp(b_noncancer_log)
//
// Interpretation:
//   b0[s]          = log tumor baseline at diploid CN
//   b_scaling[s]   = proportional dosage sensitivity
//   b_deviation[s] = linear deviation from proportional scaling
// -----------------------------------------------------------------------------

data {
  int<lower=1> N;
  array[N] int<lower=0> y;

  int<lower=2> S;
  array[N] int<lower=1, upper=S> subtype;

  vector[N] sf;
  vector<lower=0, upper=1>[N] purity;

  vector[N] dose_log;       // log(CN/2)
  vector[N] dev;            // (CN-2)/2
}

parameters {
  real b0_mean;
  vector[S] b0_offset;

  real b_scaling_mean;
  vector[S] b_scaling_offset;

  real b_dev_mean;
  vector[S] b_dev_offset;

  real b_noncancer_log;

  real<lower=1e-6> phi;
}

transformed parameters {

  vector[S] b0;
  vector[S] b_scaling;
  vector[S] b_deviation;

  vector[N] mu;

  // sum-to-zero centering
  vector[S] b0_off      = b0_offset        - mean(b0_offset);
  vector[S] scale_off   = b_scaling_offset - mean(b_scaling_offset);
  vector[S] dev_off     = b_dev_offset     - mean(b_dev_offset);

  b0          = b0_mean        + b0_off;
  b_scaling   = b_scaling_mean + scale_off;
  b_deviation = b_dev_mean     + dev_off;

  for (n in 1:N) {
    int s = subtype[n];
    real stroma_frac = 1.0 - purity[n];

    real linpred =
      b0[s]
      + dose_log[n] * b_scaling[s]
      + dev[n]      * b_deviation[s];

    linpred = fmin(linpred, 20);   // prevent overflow

    real tumor_mu =
      sf[n] * purity[n] * exp(linpred);

    real stroma_mu =
      sf[n] * stroma_frac * exp(b_noncancer_log);

    mu[n] = tumor_mu + stroma_mu;
  }
}

model {

  b0_mean          ~ normal(0, 0.5);
  b0_offset        ~ normal(0, 0.3);

  b_scaling_mean   ~ normal(0, 0.5);
  b_scaling_offset ~ normal(0, 0.3);

  b_dev_mean       ~ normal(0, 0.1);
  b_dev_offset     ~ normal(0, 0.1);

  b_noncancer_log  ~ normal(0, 0.5);

  phi ~ exponential(1);

  y ~ neg_binomial_2(mu, phi);
}

generated quantities {

  real delta_tumor0_log;
  real tumor0_fc;

  real delta_scaling;
  real delta_dev;

  vector[S] tumor_fc_2to1;
  vector[S] fracCN_2to1;

  vector[S] tumor_fc_2to3;
  vector[S] fracCN_2to3;

  vector[S] tumor_fc_2to4;
  vector[S] fracCN_2to4;

  // Net log fold-changes (numerically stable)
  vector[S] lp_2to1;  // log(tumor_fc_2to1)
  vector[S] lp_2to3;  // log(tumor_fc_2to3)
  vector[S] lp_2to4;  // log(tumor_fc_2to4)

  // Mechanistic decomposition into scaling-only and deviation-only pieces
  // lp_net = lp_scaling + lp_dev
  vector[S] lp_scaling_2to1;
  vector[S] lp_dev_2to1;

  vector[S] lp_scaling_2to3;
  vector[S] lp_dev_2to3;

  vector[S] lp_scaling_2to4;
  vector[S] lp_dev_2to4;

  // OPTIONAL: cancellation indices (signed), useful for interpretation
  // negative means deviation opposes scaling
  vector[S] cancel_index_2to1;
  vector[S] cancel_index_2to3;
  vector[S] cancel_index_2to4;

  vector[N] log_lik;

  // subtype 2 vs 1
  delta_tumor0_log = b0[2] - b0[1];
  tumor0_fc        = exp(delta_tumor0_log);

  delta_scaling = b_scaling[2] - b_scaling[1];
  delta_dev     = b_deviation[2] - b_deviation[1];

  for (s in 1:S) {
    // -------------------------
    // CN 2 -> 1 (single-copy loss)
    // dose_log = log(1/2), dev = -0.5
    // -------------------------
    lp_scaling_2to1[s] = log(1.0 / 2.0) * b_scaling[s];
    lp_dev_2to1[s]     = -0.5 * b_deviation[s];
    lp_2to1[s]         = lp_scaling_2to1[s] + lp_dev_2to1[s];

    tumor_fc_2to1[s] = exp(lp_2to1[s]);
    fracCN_2to1[s]   = tumor_fc_2to1[s] - 1.0;

    cancel_index_2to1[s] =
      lp_dev_2to1[s] / fmax(abs(lp_scaling_2to1[s]), 1e-12);

    // -------------------------
    // CN 2 -> 3 (single-copy gain)
    // dose_log = log(3/2), dev = +0.5
    // -------------------------
    lp_scaling_2to3[s] = log(3.0 / 2.0) * b_scaling[s];
    lp_dev_2to3[s]     = 0.5 * b_deviation[s];
    lp_2to3[s]         = lp_scaling_2to3[s] + lp_dev_2to3[s];

    tumor_fc_2to3[s] = exp(lp_2to3[s]);
    fracCN_2to3[s]   = tumor_fc_2to3[s] - 1.0;

    cancel_index_2to3[s] =
      lp_dev_2to3[s] / fmax(abs(lp_scaling_2to3[s]), 1e-12);

    // -------------------------
    // CN 2 -> 4 (amplification)
    // dose_log = log(4/2), dev = +1.0
    // -------------------------
    lp_scaling_2to4[s] = log(4.0 / 2.0) * b_scaling[s];
    lp_dev_2to4[s]     = 1.0 * b_deviation[s];
    lp_2to4[s]         = lp_scaling_2to4[s] + lp_dev_2to4[s];

    tumor_fc_2to4[s] = exp(lp_2to4[s]);
    fracCN_2to4[s]   = tumor_fc_2to4[s] - 1.0;

    cancel_index_2to4[s] =
      lp_dev_2to4[s] / fmax(abs(lp_scaling_2to4[s]), 1e-12);
  }

  for (n in 1:N)
    log_lik[n] = neg_binomial_2_lpmf(y[n] | mu[n], phi);
}