data {
  int<lower=1> N;
  array[N] int<lower=0> y;

  int<lower=1> K;
  array[N] int<lower=1,upper=K> covar;

  vector[N] sf;
  vector[N] purity;
  vector[N] stroma;

  vector[N] eup_equiv_cancer;
  vector[N] eup_dev_cancer;
}

parameters {
  vector<lower=0>[K] b_scaling;
  vector<lower=0>[K] b_noncancer;
  real b_deviation;

  //real<lower=0> phi;
  real<lower=1e-6, upper=1e4> phi;
}

transformed parameters {
  vector[N] mu;

  for (n in 1:N) {
    real scaling   = sf[n] * purity[n] * eup_equiv_cancer[n] * b_scaling[covar[n]];
    real noncancer = sf[n] * stroma[n] * b_noncancer[covar[n]];
    real deviation = sf[n] * purity[n] * eup_dev_cancer[n] * b_deviation;

    mu[n] = scaling + noncancer + deviation;

    // numerical safety (identity-link NB)
    if (mu[n] < 1e-6)
      mu[n] = 1e-6;
  }
}

model {
  // Priors (matching BRMS)
  b_deviation ~ normal(0, 0.2);
  b_scaling ~ lognormal(0, 1);
  b_noncancer ~ lognormal(0, 1);
  phi ~ exponential(1);
  //phi ~ lognormal(0, 1);

  y ~ neg_binomial_2(mu, phi);
}