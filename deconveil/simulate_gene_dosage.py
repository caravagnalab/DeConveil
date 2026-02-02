from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm

from deconveil.utils_fit import *
from deconveil.utils_processing import *

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, conversion, Formula
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr


def estimate_deseq2_params(counts: pd.DataFrame,
                           metadata: pd.DataFrame,
                           n_genes: int | None = None,
                           design: str = "~ 1",
                           seed: int = 1):

    deseq2 = importr("DESeq2")

    # 1) Random gene sampling
    
    if n_genes is None:
        n_genes = counts.shape[0]

    rng = np.random.default_rng(seed)
    selected_genes = rng.choice(counts.index, size=n_genes, replace=False)
    counts_sub = counts.loc[selected_genes]

    # Convert to R
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        counts_r   = conversion.py2rpy(counts_sub.astype(int))
        metadata_r = conversion.py2rpy(metadata)

    ro.globalenv["countData"] = counts_r
    ro.globalenv["colData"]   = metadata_r

    # Build DESeq2 object
    dds = deseq2.DESeqDataSetFromMatrix(
        countData=ro.globalenv["countData"],
        colData=ro.globalenv["colData"],
        design=ro.Formula(design)
    )

    dds = deseq2.DESeq(dds)
    ro.globalenv["dds"] = dds

    # 2) Extract DESeq2 internals via R helper
    
    ro.r("""
    extract_mean_disp <- function(dds) {
        env <- environment(dds@dispersionFunction)
        disp <- env$fit$model[["disps[good]"]]
        mu   <- env$fit$data[["means"]]

        # Ensure same length: subset mu to length(disp)
        if (length(mu) > length(disp)) {
            mu <- mu[1:length(disp)]
        }

        list(disp=disp, mean=mu)
    }
    """)

    res = ro.globalenv["extract_mean_disp"](dds)

    disp = np.array(res.rx2("disp"), dtype=float)
    mu   = np.array(res.rx2("mean"), dtype=float)

    # 3) Extract gene names matching disp length
    
    genes_all = list(ro.r("rownames(dds)"))

    if len(genes_all) < len(disp):
        raise ValueError(
            f"DESeq2 returned {len(disp)} dispersions but only {len(genes_all)} genes."
        )

    gene_order = genes_all[: len(disp)]

    # 4) Return matched values
    
    return mu, disp, gene_order


# Score genes by CN activity in real data
def compute_cn_gene_stats(cn: pd.DataFrame):
    vals = cn.values.astype(float)
    return pd.DataFrame({
        "mean": vals.mean(axis=1),
        "sd": vals.std(axis=1),
        "efrac": (vals != 2).mean(axis=1),
    }, index=cn.index)


# Assign genes to CN regimes by ranking, not thresholds
def assign_cn_classes_by_quantile(
    cn: pd.DataFrame,
    frac_diploid=0.40,
    frac_dig=0.30,
    frac_dsg=0.15,
    frac_dcg=0.15,
    w_mean=0.8,   # NEW: weight for CN mean shift
    w_sd=0.6,
    w_efrac=0.4,
):
    stats = compute_cn_gene_stats(cn)

    # stronger emphasis on mean CN deviation
    score = (
        w_sd    * stats["sd"]
        + w_mean * np.abs(stats["mean"] - 2.0)
        + w_efrac * stats["efrac"]
    )

    G = len(score)
    order = score.sort_values().index

    n_dip = int(frac_diploid * G)
    n_dig = int(frac_dig * G)
    n_dsg = int(frac_dsg * G)

    labels = pd.Series(index=score.index, dtype=object)
    labels.loc[order[:n_dip]] = "NEUTRAL"
    labels.loc[order[n_dip : n_dip + n_dig]] = "DIG"
    labels.loc[order[n_dip + n_dig : n_dip + n_dig + n_dsg]] = "DSG"
    labels.loc[order[n_dip + n_dig + n_dsg :]] = "DCG"
    
    return labels


def adjust_cn_to_profiles(
    cn: pd.DataFrame,
    labels: pd.Series,
    seed: int = 1,
    max_cn: int = 6,
    dsg_strength=None,
    dcg_strength=None,
    cn_heterogeneity_sd=0.20,
):
    """
    """
    rng = np.random.default_rng(seed)
    CN = cn.values.astype(float)
    G, N = CN.shape

    if dcg_strength is None:
        dcg_strength = dsg_strength

    row_index = np.arange(G)
    for i, cls in zip(row_index, labels.values):
        row = CN[i, :]
        mean_cn = row.mean()

        if cls == "NEUTRAL":
            # fully diploid
            row = np.full(N, 2.0)

        elif cls == "DIG":
            # mild deviation from 2, both up/down
            shrink = rng.uniform(0.4, 0.6)
            row = 2.0 + shrink * (row - 2.0)
            row += rng.normal(0, 0.15, size=N)

        elif cls in ("DSG", "DCG"):
            if mean_cn >= 2.0:
                # GAINS / AMPLIFICATIONS → apply strength
                strength = rng.uniform(*(dsg_strength if cls == "DSG" else dcg_strength))
                base = 2.0 + strength  # only upwards, no artificial deletions
                row = base + rng.normal(0, cn_heterogeneity_sd, size=N)
            else:
                # DELETIONS: keep real deletion pattern, just tiny noise
                row = row + rng.normal(0, 0.05, size=N)

        # safety
        CN[i, :] = np.round(np.clip(row, 0, max_cn))

    return pd.DataFrame(CN.astype(int), index=cn.index, columns=cn.columns)
    

def inject_cn_deletions(
    cn: pd.DataFrame,
    labels: pd.Series,
    del_frac_dcg: float = 0.1,
    del_frac_dsg: float = 0.05,
    frac_cn0: float = 0.3,   # fraction of deletions that are CN=0
    seed: int = 1,
):
    """
    Inject deletions into DSG and DCG genes.
    A fraction of deletions are homozygous (CN=0), others hemizygous (CN=1).
    """
    rng = np.random.default_rng(seed)
    CN = cn.values.astype(float)

    for cls, frac in [("DCG", del_frac_dcg), ("DSG", del_frac_dsg)]:
        idx = np.where(labels.values == cls)[0]
        n_del = int(frac * len(idx))
        if n_del == 0:
            continue

        del_genes = rng.choice(idx, size=n_del, replace=False)

        for g in del_genes:
            if rng.uniform() < frac_cn0:
                CN[g, :] = 0.0   # homozygous deletion
            else:
                CN[g, :] = 1.0   # hemizygous deletion

    return pd.DataFrame(CN.astype(int), index=cn.index, columns=cn.columns)


def sample_gene_parameters(
    cn_values: np.ndarray,  
    labels: np.ndarray,           # fixed gene classes (length G)
    cn_shift_log2: np.ndarray,    # c_g = log2(mean_tumor_CN / 2)
    seed: int = 1,
    max_log2fc: float = 3.0,
):
    """
    Sample gene-specific biological effects given fixed CN classes.

    Classes:
      NEUTRAL: τ = 0
      DIG    : τ large, CN-independent
      DSG    : τ ≈ 0 (CN drives naive DE)
      DCG    : τ large but compensated at RNA level (CN-aware DE)
    """

    rng = np.random.default_rng(seed)
    LOG2 = np.log(2.0)

    G = len(labels)

    # outputs
    tau = np.zeros(G)     # τ_g : CN-aware biology LFC
    beta1 = np.zeros(G)   # regression coefficient
   
    for g, cls in enumerate(labels):
        c_g = cn_shift_log2[g]
        sign = rng.choice([-1, 1])

        # NEUTRAL
        if cls == "NEUTRAL":
            tau_g = rng.normal(0, 0.05)
            #naive_g = tau_g + rng.normal(0.0, 0.05)

        # DIG — biology dominant
        elif cls == "DIG":
            sign = rng.choice([-1, 1])
            tau_g = sign * rng.uniform(0.8, 2.5)
            #naive_g = tau_g + 0.2 * c_g  # small CN tilt

        # DSG — CN-driven apparent DE
        elif cls == "DSG":
            tau_g = rng.normal(0, 0.1)
            #naive_g = sign * abs(c_g) * rng.uniform(1.0, 2.0)

        # DCG — dosage compensated
        # --------------------------
        elif cls == "DCG":
           base = -c_g
           #Enforce minimum effect size for CN-aware DE
           min_effect = 1.2   # tune: 0.6 (weak) → 1.2 (strong)
           if abs(base) < min_effect:
               base = np.sign(base if base != 0 else rng.choice([-1, 1])) * min_effect
               
           #Small imperfection in compensation
           tau_g = base + rng.normal(0, 0.1)

        else:
            raise ValueError(f"Unknown class {cls}")

        # clip for numerical safety
        tau_g = float(np.clip(tau_g, -max_log2fc, max_log2fc))
      
        tau[g] = tau_g
        beta1[g] = tau_g * LOG2   # biology-only regression effect

    return labels, beta1, tau


def cn_aware_rna_simulator(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    cn_tumor: pd.DataFrame,
    cn_normal: Optional[pd.DataFrame] = None,
    n_genes: Optional[int] = None,
    design: str = "~ 1",
    seed: int = 1,
    inject_del: bool = True,
    del_frac_dcg: Optional[float] = None,
    del_frac_dsg: Optional[float] = None,
    frac_cn0: Optional[float] = None,
    max_log2fc: float = 3.0,
    n_normal_sim: Optional[int] = None,
    n_tumor_sim: Optional[int] = None,
    bootstrap_cn: bool = True,
    seqdepth_min: float = 0.7,
    seqdepth_max: float = 1.4,
    diff_disp_range: tuple = (0.5, 2.0),
    disp_scale_norm: float = 1.0,
    disp_scale_tum: float = 1.0,
    diff_disp_frac: float = 0.3,
    w_mean=0.8,
    w_sd=0.6,
    w_efrac=0.4,
    cn_heterogeneity_sd = 0.2,
    dsg_strength=(1.8, 3.0),
    dcg_strength=(1.8, 3.0),
    null_mode: str = None,
):
    
    rng = np.random.default_rng(seed)
    LOG2 = np.log(2.0)
    eps  = 0.1

    
    # ---------------------------------------------------------
    # 1) DESeq2 baseline: mu0_g (normal mean) and dispersion
    # ---------------------------------------------------------
    mu0_arr, disp_arr, gene_order = estimate_deseq2_params(
        counts=counts,
        metadata=metadata,
        n_genes=n_genes,
        design=design,
        seed=seed,
    )

    G = len(gene_order)
    print(f"[Simulator] Using {G} genes.")

    mu0  = pd.Series(mu0_arr, index=gene_order) # shape (G,)
    disp = pd.Series(disp_arr, index=gene_order)

    # ---------------------------------------------------------
    # 2) Tumor CN: align to genes and compute CN-driven classes
    # ---------------------------------------------------------
    
    cn_tumor_sub = cn_tumor.reindex(gene_order).astype(float)
    
    if cn_tumor_sub.isna().any().any():
        raise ValueError("Tumor CN matrix missing genes.")

    # Assign classes by CN activity quantiles

    labels_series = assign_cn_classes_by_quantile(
        cn_tumor_sub,
        frac_diploid=0.50,
        frac_dig=0.30,
        frac_dsg=0.10,
        frac_dcg=0.10,
        w_mean=w_mean,   
        w_sd=w_sd,
        w_efrac=w_efrac,
    ) # pandas Series, index = genes

    
    # Adjust CN profiles per class (minimal distortion)
    
    cn_tumor_sub = adjust_cn_to_profiles(
        cn=cn_tumor_sub,
        labels=labels_series,
        seed=seed,
        max_cn=6,
        dsg_strength= dsg_strength,   
        dcg_strength=dcg_strength,
        cn_heterogeneity_sd = cn_heterogeneity_sd, 
    )
    
    # Optional deletion injection in tumors

    if inject_del:
        cn_tumor_sub = inject_cn_deletions(
            cn=cn_tumor_sub,
            labels=labels_series,
            del_frac_dcg=del_frac_dcg,
            del_frac_dsg=del_frac_dsg,
            frac_cn0=frac_cn0,
            seed=seed,
        )
    
    # ---------------------------------------------------------
    # 3) Normal CN (diploid)
    # ---------------------------------------------------------
    if cn_normal is not None:
        cn_normal_sub = cn_normal.reindex(gene_order).fillna(2)
    else:
        cn_normal_sub = pd.DataFrame(
            2,
            index=gene_order,
            columns=[f"N{i+1}" for i in range(cn_tumor_sub.shape[1])]
        )
    
    # ---------------------------------------------------------
    # 4) Bootstrap CN samples to desired cohort size
    # ---------------------------------------------------------
    if bootstrap_cn:
        if n_normal_sim is not None:
            cols = cn_normal_sub.columns
            idx  = rng.choice(cols, size=n_normal_sim, replace=True)
            cn_normal_sub = cn_normal_sub.loc[:, idx]

        if n_tumor_sim is not None:
            cols = cn_tumor_sub.columns
            idx  = rng.choice(cols, size=n_tumor_sim, replace=True)
            cn_tumor_sub = cn_tumor_sub.loc[:, idx]

    n_normal = cn_normal_sub.shape[1]
    n_tumor  = cn_tumor_sub.shape[1]

   
    # ---------------------------------------------------------
    # 5) CN shift term c_g = log2(mean_tumor_CN/2 + 0.1)
    # ---------------------------------------------------------
    cn_tum_mean = cn_tumor_sub.mean(axis=1).values.astype(float)  # (G,)
    cn_shift_log2 = np.log((cn_tum_mean / 2.0 + 0.1)) / LOG2      # (G,)
    
    # Convert labels to array aligned with gene_order
    labels = labels_series.loc[gene_order].to_numpy(dtype=object)

    # ---------------------------------------------------------
    # 6) Null simulation - CN ≈ 2 in both groups, no β₁
    # ---------------------------------------------------------
    
    if null_mode == "pure_null":
        cn_tumor_sub.loc[:, :] = 2.0
    if cn_normal_sub is not None:
        cn_normal_sub.loc[:, :] = 2.0

    # ---------------------------------------------------------
    # 6) Sample biology-only effects sampling
    # ---------------------------------------------------------
    if null_mode is None:
        labels, beta1, tau = sample_gene_parameters(
            cn_values = cn_tumor_sub.values,
            labels=labels,
            cn_shift_log2 = cn_shift_log2,
            seed = seed,
            max_log2fc = max_log2fc
        )

    else:
        # For null scenarios, NO biology: θ = 0, β₁ = 0
        G = len(gene_order)
        tau = np.zeros(G, dtype=float)
        beta1 = np.zeros(G, dtype=float)

    # ---------------------------------------------------------
    # 7) Build μ using CN scaling (single generative model)
    # ---------------------------------------------------------

    cn_all = pd.concat([cn_normal_sub, cn_tumor_sub], axis=1)
    CN = cn_all.values.astype(float)   # (G × N)
    samples = cn_all.columns.tolist()
    N = CN.shape[1]

    mu0_vals = mu0.values

    cn_scaled = CN.copy()

    cn_scaled = CN / 2.0                      # no DCG special case
    cn_norm_scaled = cn_scaled[:, :n_normal]
    norm_factor = np.clip(cn_norm_scaled.mean(axis=1), 1e-12, None)

    # Intercept β₀,g chosen so that the *average* normal mean matches mu0_g
    
    beta0 = np.log(mu0_vals + eps) - np.log(norm_factor)

    # Condition vector (0 = normal, 1 = tumor)
    cond = np.concatenate([np.zeros(n_normal, dtype=float),
                           np.ones(n_tumor, dtype=float)])  # (N,)

    beta0 = beta0.astype(float)
    beta1 = beta1.astype(float)

    log_mu = beta0[:, None] + beta1[:, None] * cond[None, :] + np.log(cn_scaled + eps)
    mu = np.exp(log_mu)  # (G,N)

    # ---------------------------------------------------------
    # 8) Sequencing-depth factors (library sizes)
    # ---------------------------------------------------------
    depth_factors = rng.uniform(seqdepth_min, seqdepth_max, size=N)
    mu_depth = mu * depth_factors[None, :]

    # ---------------------------------------------------------
    # 9) Differential dispersion per condition
    # ---------------------------------------------------------
    
    phi_norm = disp.values.astype(float) * disp_scale_norm
    phi_tum  = disp.values.astype(float) * disp_scale_tum

    if diff_disp_frac > 0:
        n_diff = int(diff_disp_frac * G)
        diff_idx = rng.choice(G, size=n_diff, replace=False)
        low, high = diff_disp_range
        phi_tum[diff_idx] *= rng.uniform(low, high, size=n_diff)

    phi_mat = np.concatenate(
        [np.tile(phi_norm[:, None], (1, n_normal)),
         np.tile(phi_tum[:, None], (1, n_tumor))],
        axis=1,
    )
    phi_mat = np.clip(phi_mat, 1e-6, 10.0)

    size_mat = 1.0 / phi_mat
    p_mat = size_mat / (size_mat + mu_depth)

    # ---------------------------------------------------------
    # 10) Negative Binomial sampling of RNA counts
    # ---------------------------------------------------------
    counts_sim = nbinom.rvs(size_mat, p_mat, random_state=rng)
    counts_df = pd.DataFrame(counts_sim, index=gene_order, columns=samples)
    cn_df = pd.DataFrame(CN.astype(int), index=gene_order, columns=samples)

    # ---------------------------------------------------------
    # 11) Empirical LFCs from the model means
    # ---------------------------------------------------------

    mu_norm = mu_depth[:, :n_normal].mean(axis=1)
    mu_tum = mu_depth[:, n_normal:].mean(axis=1)

    empirical_naive = np.log2((mu_tum + 1e-12) / (mu_norm + 1e-12))

    CN_norm = CN[:, :n_normal]
    CN_tum = CN[:, n_normal:]
    mu_norm_div = (mu_depth[:, :n_normal] / (CN_norm / 2.0 + eps)).mean(axis=1)
    mu_tum_div = (mu_depth[:, n_normal:] / (CN_tum / 2.0 + eps)).mean(axis=1)
    empirical_aware = np.log2((mu_tum_div + 1e-12) / (mu_norm_div + 1e-12))

    # "designed" truths
    truth_lfc_aware = tau                   # tau_g
    truth_lfc_naive = tau+ cn_shift_log2    # tau_g + c_g

    # ---------------------------------------------------------
    # 12) Truth and DE labels
    # ---------------------------------------------------------

    # For CN-aware, biology-only truth is theta (what beta1 encodes).
    # For CN-naive, we take the *generative* LFC to be the empirical_naive
    # derived from mu_depth (not the design helper `naive_design`).

    EPS_TRUTH = 0.25 # 0.0 strict null

    truth_df = pd.DataFrame(
        {
            "class": labels,
            "beta1": beta1,
            "cn_ratio_log2": cn_shift_log2,
            "truth_log2FC_aware": truth_lfc_aware,
            "truth_log2FC_naive": truth_lfc_naive,  
            "empirical_log2FC_naive": empirical_naive,
            "empirical_log2FC_aware": empirical_aware,
        },
        index=gene_order,
    )

    if null_mode is not None:
        truth_df["DE_truth"] = False  # global null
    else:
        # usual non-null DE logic
        truth_df["DE_truth"] = np.abs(truth_df["truth_log2FC_aware"]) > EPS_TRUTH  


    # ---------------------------------------------------------
    # 13) Metadata and dispersions
    # ---------------------------------------------------------
    
    meta = pd.DataFrame(
        {"condition": ["normal"]*n_normal + ["tumor"]*n_tumor},
        index=cn_all.columns
    )

    # return both normal and tumor dispersions
    disp_df = pd.DataFrame(
        {"disp_norm": phi_norm, "disp_tumor": phi_tum}, index=gene_order
    )

    return {
        "counts": counts_df,
        "CN": cn_df,
        "truth": truth_df,
        "mu": mu_depth,
        "metadata": meta,
        "trueDispersions": disp_df,
        "size_factors": depth_factors
    }