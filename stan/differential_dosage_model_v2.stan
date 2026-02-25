// -----------------------------------------------------------------------------
// Gene-by-gene subtype-aware CN-expression model (log link)
// Improves regularization across subtypes - estimated SD from data with shrinkage toward zero SD
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
  // global means
  real b0_mean;
  real b_scaling_mean;
  real b_dev_mean;

  // non-centered subtype offsets
  vector[S] z_b0;
  vector[S] z_b_scaling;
  vector[S] z_b_dev;

  // hierarchical scales (between-subtype SDs)
  real<lower=0> tau_b0;
  real<lower=0> tau_b_scaling;
  real<lower=0> tau_b_dev;

  real b_noncancer_log;

  real log_phi;   // as in the VI-friendly version
}

transformed parameters {
  vector[S] b0;
  vector[S] b_scaling;
  vector[S] b_deviation;

  vector[S] b0_off;
  vector[S] scale_off;
  vector[S] dev_off;

  vector[N] log_mu;
  real<lower=0> phi = exp(log_phi);

  // sum-to-zero centered offsets via non-centered parameterization
  b0_off      = tau_b0       * (z_b0       - mean(z_b0));
  scale_off   = tau_b_scaling* (z_b_scaling- mean(z_b_scaling));
  dev_off     = tau_b_dev    * (z_b_dev    - mean(z_b_dev));

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

    real log_tumor_mu =
      log(sf[n]) + log(purity[n]) + linpred;

    real log_stroma_mu =
      log(sf[n]) + log(stroma_frac) + b_noncancer_log;

    log_mu[n] = log_sum_exp(log_tumor_mu, log_stroma_mu);
  }
}

model {
  // global means (same as before or similar)
  b0_mean        ~ normal(0, 0.5);
  b_scaling_mean ~ normal(0, 0.5);
  b_dev_mean     ~ normal(0, 0.1);

  // hierarchical scales (half-normal-ish)
  tau_b0        ~ normal(0, 0.3);   // half-normal via <lower=0>
  tau_b_scaling ~ normal(0, 0.3);
  tau_b_dev     ~ normal(0, 0.1);

  // non-centered latent z's
  z_b0        ~ normal(0, 1);
  z_b_scaling ~ normal(0, 1);
  z_b_dev     ~ normal(0, 1);

  b_noncancer_log ~ normal(0, 0.5);

  log_phi ~ normal(0, 1);

  y ~ neg_binomial_2_log(log_mu, phi);
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
    log_lik[n] = neg_binomial_2_log_lpmf(y[n] | log_mu[n], phi);
}