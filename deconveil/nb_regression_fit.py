from cmdstanpy import CmdStanModel
import numpy as np
import pandas as pd
import scipy.stats as st


def fit_one_gene(
    gene_df: pd.DataFrame,
    model: CmdStanModel,
    gene: str | None = None,      # optional convenience
    cna: str = "all",             # "amp" | "del" | "all"
    et: float = 0.15,
    min_aneup: int = 5,
    min_unique_counts: int = 5,
    min_cn_abs_sum: float = 1.0,  # identifiability filter for cna="all"
    chains: int = 4,
    iter_warmup: int = 1000,
    iter_sampling: int = 1000,
    seed: int = 1,
    show_progress: bool = False,
    adapt_delta: float = 0.99,
    max_treedepth: int = 15,
):
    """
    Fit Stan NB regression model for a single gene.
    Expects gene_df to be filtered to one gene OR pass gene and full df.

    Required columns in gene_df:
      gene, expr, copies, purity, stroma, sf, eup_dev_cancer, eup_equiv_cancer
    Optional columns:
      covar (if missing -> set to 'ALL')
    """
    # subset to one gene if gene provided and gene_df contains multiple genes
    df = gene_df.copy()
    if gene is not None and "gene" in df.columns and df["gene"].nunique() > 1:
        df = df.loc[df["gene"] == gene].copy()
    if gene is None and "gene" in df.columns and df["gene"].nunique() == 1:
        gene = str(df["gene"].iloc[0])

    if df.empty:
        return {"status": "skipped", "gene": gene, "reason": "no_rows_for_gene"}

    required = {"expr","copies","purity","stroma","sf","eup_dev_cancer","eup_equiv_cancer"}
    missing = required - set(df.columns)
    if missing:
        return {"status": "error", "gene": gene, "reason": f"missing_columns: {sorted(missing)}"}

    # CNA subset (optional)
    if cna == "amp":
        df = df[df["copies"] > (2 - et)]
    elif cna == "del":
        df = df[df["copies"] < (2 + et)]
    elif cna == "all":
        pass
    else:
        raise ValueError("cna must be 'amp', 'del', or 'all'")

    if df.empty:
        return {"status": "skipped", "gene": gene, "reason": "no_samples_after_cna_filter"}

    # basic data QC
    df = df.dropna(subset=["expr","sf","purity","stroma","eup_dev_cancer","eup_equiv_cancer"])
    if df.empty:
        return {"status": "skipped", "gene": gene, "reason": "all_na_after_dropna"}

    if (df["expr"] < 0).any():
        return {"status": "error", "gene": gene, "reason": "negative_counts"}

    if not df["purity"].between(0, 1).all():
        return {"status": "error", "gene": gene, "reason": "purity_out_of_bounds"}

    if not (df["sf"] > 0).all():
        return {"status": "error", "gene": gene, "reason": "nonpositive_sf"}

    if not (df["eup_equiv_cancer"] > 0).all():
        return {"status": "error", "gene": gene, "reason": "nonpositive_eup_equiv_cancer"}

    # skip degenerate genes early (prevents phi->inf + treedepth explosions)
    if df["expr"].nunique() < min_unique_counts:
        return {"status": "skipped", "gene": gene, "reason": "too_few_unique_counts"}

    # aneuploid count check (use copies as cancer_copies analogue)
    n_aneup = int((np.abs(df["copies"].astype(float) - 2.0) > (1.0 - et)).sum())
    if n_aneup < min_aneup or (df["expr"] == 0).all():
        return {"status": "skipped", "gene": gene, "n_aneup": n_aneup, "reason": "low_aneup_or_all_zero"}

    # identifiability check for cna="all": need some CN deviation mass
    if cna == "all" and df["eup_dev_cancer"].abs().sum() < min_cn_abs_sum:
        return {"status": "skipped", "gene": gene, "n_aneup": n_aneup, "reason": "too_little_cn_variation"}

    # within one cohort: enforce K=1, covar='ALL'
    df["covar"] = "ALL"
    covar_levels = ["ALL"]
    covar_idx = np.ones(len(df), dtype=int)

    # per-gene sf scaling 
    mean_expr = float(df["expr"].mean())
    df["sf_scaled"] = df["sf"].astype(float) * mean_expr

    stan_data = {
        "N": int(len(df)),
        "y": df["expr"].astype(int).to_numpy(),
        "K": 1,
        "covar": covar_idx,
        "sf": df["sf_scaled"].to_numpy(dtype=float),
        "purity": df["purity"].to_numpy(dtype=float),
        "stroma": df["stroma"].to_numpy(dtype=float),
        "eup_equiv_cancer": df["eup_equiv_cancer"].to_numpy(dtype=float),
        "eup_dev_cancer": df["eup_dev_cancer"].to_numpy(dtype=float),
    }

    rng = np.random.default_rng(seed)
    init = {
        "b_scaling": rng.uniform(0.5, 1.5, size=1).tolist(),
        "b_noncancer": rng.uniform(0.5, 1.5, size=1).tolist(),
        "b_deviation": 0.0,
        "phi": 1.0,
    }

    fit = model.sample(
        data=stan_data,
        chains=chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        seed=seed,
        inits=init,
        show_progress=show_progress,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
    )

    draws = fit.draws_pd()

    # extract posterior summaries
   
    scaling_col = next((c for c in draws.columns if c.startswith("b_scaling")), None)
    if scaling_col is None:
        return {"status": "error", "gene": gene, "reason": "missing_b_scaling_draws"}

    if "b_deviation" not in draws.columns:
        return {"status": "error", "gene": gene, "reason": "missing_b_deviation_draws"}

    phi_col = "phi" if "phi" in draws.columns else None

    b_scaling = draws[scaling_col].to_numpy()
    b_dev = draws["b_deviation"].to_numpy()

    # z and p 
    z_comp = float(b_dev.mean() / b_dev.std(ddof=1))
    p_value = float(2.0 * (1.0 - st.norm.cdf(abs(z_comp))))

    summ = fit.summary()

    # robust extraction of Rhat / ESS
    rhat_col = next((c for c in ["R_hat", "Rhat"] if c in summ.columns), None)
    ess_col  = next((c for c in ["Ess_bulk", "ESS_bulk", "N_Eff", "Ess"] if c in summ.columns), None)

    rhat_dev = float(summ.loc["b_deviation", rhat_col]) if (rhat_col and "b_deviation" in summ.index) else np.nan
    ess_dev  = float(summ.loc["b_deviation", ess_col]) if (ess_col and "b_deviation" in summ.index) else np.nan

    # mark borderline fits as warn (you can filter later)
    status = "ok"
    if (not np.isnan(rhat_dev) and rhat_dev > 1.05) or (not np.isnan(ess_dev) and ess_dev < 200):
        status = "warn"

    out = {
        "status": status,
        "gene": gene,
        "N": int(len(df)),
        "n_aneup": n_aneup,
        "cna": cna,
        # posterior summaries
        "mean_b_scaling": float(b_scaling.mean()),
        "sd_b_scaling": float(b_scaling.std(ddof=1)),
        "mean_b_deviation": float(b_dev.mean()),
        "sd_b_deviation": float(b_dev.std(ddof=1)),
        "z_comp": z_comp,
        "p_value": p_value,
        # optional
        "mean_phi": float(draws[phi_col].mean()) if phi_col else np.nan,
        "Rhat_b_deviation": rhat_dev,
        "ess_b_deviation": ess_dev,
        "covar_levels": covar_levels,
    }
    return out


def postprocess_nb_results(
    res_df: pd.DataFrame,
    alpha: float = 0.05,
    comp_thr: float = 0.5,
    min_aneup_frac: float = 0.30,   # >=30% aneuploid samples
    min_scaling: float = 1e-6,
    require_scaling_for_dsg: bool = True,
    warn_suffix: str | None = None,  # e.g. "_LOWCONF" to tag warn calls
) -> pd.DataFrame:
    """
    Post-process Stan NB regression results.

    Expected columns in `res_df`:
      - status, gene, N, n_aneup, cna
      - mean_b_scaling, sd_b_scaling
      - mean_b_deviation, sd_b_deviation
      - z_comp, p_value, mean_phi
      - Rhat_b_deviation, ess_b_deviation, covar_levels

    Adds:
      - adj_p       : BH-FDR over usable genes (status in {'ok','warn'})
      - aneup_frac  : n_aneup / N (CN informativeness proxy)
      - comp_score  : normalized deviation  = mean_b_deviation / mean_b_scaling
      - signed_comp : sign-normalized comp_score (flip for 'del', keep for 'amp'/'all')
      - shrunk_comp : comp_score shrunk toward 0 by (1 - p_value)  [orthogonal score]
      - label_nb    : {'DSG','DCG','HYPER','OTHER','SKIP','ERROR'}

    Notes:
      - Without a per-gene CN direction summary, signed_comp is only partially
        identifiable for cna='all'; here we flip sign for 'del' fits and leave
        'amp'/'all' unchanged.
      - DCG/HYPER require both sufficient magnitude (|signed_comp| >= comp_thr) 
      AND statistical significance (p_value <= alpha).
      - DSG requires non-significant deviation (p_value > alpha) and small magnitude
        |signed_comp| <= comp_thr, with sufficient CN signal.
    """

    df = res_df.copy()

    # required columns check 
    required = {
        "status", "gene", "N", "n_aneup", "cna", "p_value",
        "mean_b_scaling", "mean_b_deviation"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # usable rows: include ok + warn 
    status_lower = df["status"].astype(str).str.lower()
    usable = status_lower.isin(["ok", "warn"]) & df["p_value"].notna()

    # default labels 
    df["label_nb"] = "OTHER"
    df.loc[status_lower.isin(["skip", "skipped"]), "label_nb"] = "SKIP"
    df.loc[status_lower.isin(["error", "failed", "fail"]), "label_nb"] = "ERROR"

    # FDR across usable genes 
    df["adj_p"] = np.nan
    if usable.any():
        df.loc[usable, "adj_p"] = bh_fdr(
            df.loc[usable, "p_value"].to_numpy(dtype=float)
        )

    # CN informativeness proxy (fraction aneuploid) 
    df["aneup_frac"] = np.nan
    df.loc[usable, "aneup_frac"] = (
        df.loc[usable, "n_aneup"] / df.loc[usable, "N"]
    ).astype(float)

    # compute compensation scores 
    df["comp_score"] = np.nan
    df["signed_comp"] = np.nan
    df["shrunk_comp"] = np.nan  # normalized + shrunk score (orthogonal summary)

    ok2 = (
        usable
        & df["mean_b_scaling"].notna()
        & df["mean_b_deviation"].notna()
        & (df["mean_b_scaling"].abs() > min_scaling)
        & df["aneup_frac"].notna()
        & (df["aneup_frac"] >= min_aneup_frac)
    )

    # Normalized deviation: beta2* = mean_b_deviation / mean_b_scaling
    df.loc[ok2, "comp_score"] = (
        df.loc[ok2, "mean_b_deviation"] / df.loc[ok2, "mean_b_scaling"]
    )

    # Sign-normalize by CNA type if available: del -> flip sign
    cna_vals = df.loc[ok2, "cna"].astype(str).str.lower()
    sign = np.ones(len(cna_vals), dtype=float)
    sign[cna_vals == "del"] = -1.0    # treat deletions as negative direction
    # amp/all remain +1
    df.loc[ok2, "signed_comp"] = (
        df.loc[ok2, "comp_score"].to_numpy(dtype=float) * sign
    )

    # Shrink normalized deviation by pseudo-p (here: raw p_value)
    # s_g ≈ beta2*_g * (1 - p_g)
    pseudo_p = df.loc[ok2, "p_value"].astype(float).clip(0.0, 1.0)
    df.loc[ok2, "shrunk_comp"] = df.loc[ok2, "comp_score"] * (1.0 - pseudo_p)

    # classification gates 
    # Significant deviation (
    is_sig = ok2 & (df["p_value"] <= alpha)
    is_nonsig = ok2 & (df["p_value"] > alpha)

    # Optional: require non-trivial scaling to call DSG
    scaling_ok = ok2
    if require_scaling_for_dsg:
        scaling_ok = ok2 & (df["mean_b_scaling"] > 0.3)

    # DCG: significant negative deviation (compensation)
    #df.loc[is_sig & (df["shrunk_comp"] <= -comp_thr), "label_nb"] = "DCG"
    df.loc[df["shrunk_comp"] <= -comp_thr, "label_nb"] = "DCG"

    # HYPER: significant positive deviation
    #df.loc[is_sig & (df["signed_comp"] >= comp_thr), "label_nb"] = "HYPER"
    df.loc[df["shrunk_comp"] >= comp_thr, "label_nb"] = "HYPER"

    # DSG: non-significant deviation AND near CN-proportional expectation
    df.loc[
        #is_nonsig & scaling_ok & (df["shrunk_comp"].abs() <= comp_thr),
        scaling_ok & (df["shrunk_comp"].abs() <= comp_thr),
        "label_nb"
    ] = "DSG"

    # optional: tag warn calls as low confidence 
    if warn_suffix:
        is_warn = status_lower.eq("warn")
        mask = is_warn & df["label_nb"].isin(["DSG", "DCG", "HYPER"])
        df.loc[mask, "label_nb"] = df.loc[mask, "label_nb"] + warn_suffix

    return df