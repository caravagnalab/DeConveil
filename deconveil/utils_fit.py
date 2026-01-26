import os
import multiprocessing
import warnings
from math import ceil, floor
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union, cast, Dict, Any

import numpy as np
import pandas as pd
from scipy.linalg import solve  # type: ignore
from scipy.optimize import minimize  # type: ignore
from scipy.special import gammaln  # type: ignore
from scipy.special import polygamma  # type: ignore
from scipy.stats import norm  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

from deconveil.grid_search import grid_fit_beta
from pydeseq2.utils import fit_alpha_mle
from pydeseq2.utils import get_num_processes
from pydeseq2.grid_search import grid_fit_alpha
from pydeseq2.grid_search import grid_fit_shrink_beta

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, conversion, Formula
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr


def irls_glm(
    counts: np.ndarray,
    cnv: np.ndarray,
    size_factors: np.ndarray,
    design_matrix: np.ndarray,
    disp: float,
    min_mu: float = 0.5,
    beta_tol: float = 1e-8,
    min_beta: float = -30,
    max_beta: float = 30,
    optimizer: Literal["BFGS", "L-BFGS-B"] = "L-BFGS-B",
    maxiter: int = 250,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:

    assert optimizer in ["BFGS", "L-BFGS-B"]
    
    num_vars = design_matrix.shape[1]
    X = design_matrix
    
    # if full rank, estimate initial betas for IRLS below
    if np.linalg.matrix_rank(X) == num_vars:
        Q, R = np.linalg.qr(X)
        eps = 1e-8
        cnv = np.where(cnv == 0, eps, cnv)
        y = np.log((counts / cnv) / size_factors + 0.1)
        beta_init = solve(R, Q.T @ y)
        beta = beta_init

    else:  # Initialise intercept with log base mean
        beta_init = np.zeros(num_vars)
        beta_init[0] = np.log((counts / cnv) / size_factors).mean()
        beta = beta_init
        
    dev = 1000.0
    dev_ratio = 1.0

    ridge_factor = np.diag(np.repeat(1e-6, num_vars))
    mu = np.maximum(cnv * size_factors * np.exp(np.clip(X @ beta, -30, 30)), min_mu)
    
    converged = True
    i = 0
    while dev_ratio > beta_tol:
        W = mu / (1.0 + mu * disp)
        z = np.log((mu / cnv) / size_factors) + (counts - mu) / mu
        H = (X.T * W) @ X + ridge_factor
        beta_hat = solve(H, X.T @ (W * z), assume_a="pos")
        i += 1

        if sum(np.abs(beta_hat) > max_beta) > 0 or i >= maxiter:
            # If IRLS starts diverging, use L-BFGS-B
            def f(beta: np.ndarray) -> float:
                # closure to minimize
                mu_ = np.maximum(cnv * size_factors * np.exp(np.clip(X @ beta, -30, 30)), min_mu)
                
                return nb_nll(counts, mu_, disp) + 0.5 * (ridge_factor @ beta**2).sum()

            def df(beta: np.ndarray) -> np.ndarray:
                mu_ = np.maximum(cnv * size_factors * np.exp(np.clip(X @ beta, -30, 30)), min_mu)
                return (
                    -X.T @ counts
                    + ((1 / disp + counts) * mu_ / (1 / disp + mu_)) @ X
                    + ridge_factor @ beta
                )

            res = minimize(
                f,
                beta_init,
                jac=df,
                method=optimizer,
                bounds=(
                    [(min_beta, max_beta)] * num_vars
                    if optimizer == "L-BFGS-B"
                    else None
                ),
            )
            
            beta = res.x
            mu = np.maximum(cnv * size_factors * np.exp(np.clip(X @ beta, -30, 30)), min_mu)
            converged = res.success

        beta = beta_hat
        mu = np.maximum(cnv * size_factors * np.exp(np.clip(X @ beta, -30, 30)), min_mu)
        
        # Compute deviation
        old_dev = dev
        # Replaced deviation with -2 * nll, as in the R code
        dev = -2 * nb_nll(counts, mu, disp)
        dev_ratio = np.abs(dev - old_dev) / (np.abs(dev) + 0.1)

    # Compute H diagonal (useful for Cook distance outlier filtering)
    W = mu / (1.0 + mu * disp)
    W_sq = np.sqrt(W)
    XtWX = (X.T * W) @ X + ridge_factor
    H = W_sq * np.diag(X @ np.linalg.inv(XtWX) @ X.T) * W_sq
    
    # Return an UNthresholded mu 
    # Previous quantities are estimated with a threshold though
    mu = np.maximum(cnv * size_factors * np.exp(np.clip(X @ beta, -30, 30)), min_mu)
    
    return beta, mu, H, converged


def fit_lin_mu(
    counts: np.ndarray,
    size_factors: np.ndarray,
    design_matrix: np.ndarray,
    min_mu: float = 0.5,
) -> np.ndarray:
    """Estimate mean of negative binomial model using a linear regression.

    Used to initialize genewise dispersion models.

    Parameters
    ----------
    counts : ndarray
        Raw counts for a given gene.

    size_factors : ndarray
        Sample-wise scaling factors (obtained from median-of-ratios).

    design_matrix : ndarray
        Design matrix.

    min_mu : float
        Lower threshold for fitted means, for numerical stability. (default: ``0.5``).

    Returns
    -------
    ndarray
        Estimated mean.
    """
    reg = LinearRegression(fit_intercept=False)
    reg.fit(design_matrix, counts / size_factors)
    mu_hat = size_factors * reg.predict(design_matrix)
    # Threshold mu_hat as 1/mu_hat will be used later on.
    return np.maximum(mu_hat, min_mu)


def fit_rough_dispersions(
    normed_counts: np.ndarray, design_matrix: pd.DataFrame
) -> np.ndarray:
    """Rough dispersion estimates from linear model, as per the R code.

    Used as initial estimates in :meth:`DeseqDataSet.fit_genewise_dispersions()
    <pydeseq2.dds.DeseqDataSet.fit_genewise_dispersions>`.

    Parameters
    ----------
    normed_counts : ndarray
        Array of deseq2-normalized read counts. Rows: samples, columns: genes.

    design_matrix : pandas.DataFrame
        A DataFrame with experiment design information (to split cohorts).
        Indexed by sample barcodes. Unexpanded, *with* intercept.

    Returns
    -------
    ndarray
        Estimated dispersion parameter for each gene.
    """
    num_samples, num_vars = design_matrix.shape
    # This method is only possible when num_samples > num_vars.
    # If this is not the case, throw an error.
    if num_samples == num_vars:
        raise ValueError(
            "The number of samples and the number of design variables are "
            "equal, i.e., there are no replicates to estimate the "
            "dispersion. Please use a design with fewer variables."
        )

    reg = LinearRegression(fit_intercept=False)
    reg.fit(design_matrix, normed_counts)
    y_hat = reg.predict(design_matrix)
    y_hat = np.maximum(y_hat, 1)
    alpha_rde = (
        ((normed_counts - y_hat) ** 2 - y_hat) / ((num_samples - num_vars) * y_hat**2)
    ).sum(0)
    return np.maximum(alpha_rde, 0)


def fit_moments_dispersions2(
    normed_counts: np.ndarray, size_factors: np.ndarray
) -> np.ndarray:
    """Dispersion estimates based on moments, as per the R code.

    Used as initial estimates in :meth:`DeseqDataSet.fit_genewise_dispersions()
    <pydeseq2.dds.DeseqDataSet.fit_genewise_dispersions>`.

    Parameters
    ----------
    normed_counts : ndarray
        Array of deseq2-normalized read counts. Rows: samples, columns: genes.

    size_factors : ndarray
        DESeq2 normalization factors.

    Returns
    -------
    ndarray
        Estimated dispersion parameter for each gene.
    """
    # Exclude genes with all zeroes
    #normed_counts = normed_counts[:, ~(normed_counts == 0).all(axis=0)]
    is_all_zero = (normed_counts == 0).all(axis=0)
    # if DataFrame -> Series; if ndarray -> ndarray
    mask = ~np.asarray(is_all_zero)
    if hasattr(normed_counts, "loc"):
        normed_counts = normed_counts.loc[:, mask]
    else:
        normed_counts = normed_counts[:, mask]
    #mean inverse size factor
    s_mean_inv = (1 /size_factors).mean()
    mu = normed_counts.mean(0)
    sigma = normed_counts.var(0, ddof=1)
    # ddof=1 is to use an unbiased estimator, as in R
    # NaN (variance = 0) are replaced with 0s
    return np.nan_to_num((sigma - s_mean_inv * mu) / mu**2)


def nb_nll(
    counts: np.ndarray, mu: np.ndarray, alpha: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    r"""Neg log-likelihood of a negative binomial of parameters ``mu`` and ``alpha``.

    Mathematically, if ``counts`` is a vector of counting entries :math:`y_i`
    then the likelihood of each entry :math:`y_i` to be drawn from a negative
    binomial :math:`NB(\mu, \alpha)` is [1]

    .. math::
        p(y_i | \mu, \alpha) = \frac{\Gamma(y_i + \alpha^{-1})}{
            \Gamma(y_i + 1)\Gamma(\alpha^{-1})
        }
        \left(\frac{1}{1 + \alpha \mu} \right)^{1/\alpha}
        \left(\frac{\mu}{\alpha^{-1} + \mu} \right)^{y_i}

    As a consequence, assuming there are :math:`n` entries,
    the total negative log-likelihood for ``counts`` is

    .. math::
        \ell(\mu, \alpha) = \frac{n}{\alpha} \log(\alpha) +
            \sum_i \left \lbrace
            - \log \left( \frac{\Gamma(y_i + \alpha^{-1})}{
            \Gamma(y_i + 1)\Gamma(\alpha^{-1})
        } \right)
        + (\alpha^{-1} + y_i) \log (\alpha^{-1} + \mu)
        - y_i \log \mu
            \right \rbrace

    This is implemented in this function.

    Parameters
    ----------
    counts : ndarray
        Observations.

    mu : ndarray
        Mean of the distribution :math:`\mu`.

    alpha : float or ndarray
        Dispersion of the distribution :math:`\alpha`,
        s.t. the variance is :math:`\mu + \alpha \mu^2`.

    Returns
    -------
    float or ndarray
        Negative log likelihood of the observations counts
        following :math:`NB(\mu, \alpha)`.

    Notes
    -----
    [1] https://en.wikipedia.org/wiki/Negative_binomial_distribution
    """
    n = len(counts)
    alpha_neg1 = 1 / alpha
    logbinom = gammaln(counts + alpha_neg1) - gammaln(counts + 1) - gammaln(alpha_neg1)
    if hasattr(alpha, "__len__") and len(alpha) > 1:
        return (
            alpha_neg1 * np.log(alpha)
            - logbinom
            + (counts + alpha_neg1) * np.log(mu + alpha_neg1)
            - (counts * np.log(mu))
        ).sum(0)
    else:
        return (
            n * alpha_neg1 * np.log(alpha)
            + (
                -logbinom
                + (counts + alpha_neg1) * np.log(alpha_neg1 + mu)
                - counts * np.log(mu)
            ).sum()
        )


def nbinomGLM(
    design_matrix: np.ndarray,
    counts: np.ndarray,
    cnv: np.ndarray,
    size: np.ndarray,
    offset: np.ndarray,
    prior_no_shrink_scale: float,
    prior_scale: float,
    optimizer="L-BFGS-B",
    shrink_index: int = 1,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Fit a negative binomial MAP LFC using an apeGLM prior.

    Only the LFC is shrinked, and not the intercept.

    Parameters
    ----------
    design_matrix : ndarray
        Design matrix.

    counts : ndarray
        Raw counts.

    size : ndarray
        Size parameter of NB family (inverse of dispersion).

    offset : ndarray
        Natural logarithm of size factor.

    prior_no_shrink_scale : float
        Prior variance for the intercept.

    prior_scale : float
        Prior variance for the LFC parameter.

    optimizer : str
        Optimizing method to use in case IRLS starts diverging.
        Accepted values: 'L-BFGS-B', 'BFGS' or 'Newton-CG'. (default: ``'Newton-CG'``).

    shrink_index : int
        Index of the LFC coordinate to shrink. (default: ``1``).

    Returns
    -------
    beta: ndarray
        2-element array, containing the intercept (first) and the LFC (second).

    inv_hessian: ndarray
        Inverse of the Hessian of the objective at the estimated MAP LFC.

    converged: bool
        Whether L-BFGS-B converged.
    """
    num_vars = design_matrix.shape[-1]

    shrink_mask = np.zeros(num_vars)
    shrink_mask[shrink_index] = 1
    no_shrink_mask = np.ones(num_vars) - shrink_mask

    beta_init = np.ones(num_vars) * 0.1 * (-1) ** (np.arange(num_vars))

    # Set optimization scale
    scale_cnst = nbinomFn(
        np.zeros(num_vars),
        design_matrix,
        counts,
        cnv,
        size,
        offset,
        prior_no_shrink_scale,
        prior_scale,
        shrink_index,
    )
    scale_cnst = np.maximum(scale_cnst, 1)

    def f(beta: np.ndarray, cnst: float = scale_cnst) -> float:
        # Function to optimize
        return (
            nbinomFn(
                beta,
                design_matrix,
                counts,
                cnv,
                size,
                offset,
                prior_no_shrink_scale,
                prior_scale,
                shrink_index,
            )
            / cnst
        )

    def df(beta: np.ndarray, cnst: float = scale_cnst) -> np.ndarray:
        # Gradient of the function to optimize
        xbeta = design_matrix @ beta
        d_neg_prior = (
            beta * no_shrink_mask / prior_no_shrink_scale**2
            + 2 * beta * shrink_mask / (prior_scale**2 + beta[shrink_index] ** 2),
        )
        d_nll = (
            counts - (counts + size) / (1 + size * np.exp(-xbeta - offset - cnv))
        ) @ design_matrix
            
        return (d_neg_prior - d_nll) / cnst

    def ddf(beta: np.ndarray, cnst: float = scale_cnst) -> np.ndarray:
        # Hessian of the function to optimize
        # Note: will only work if there is a single shrink index
        xbeta = design_matrix @ beta
        exp_xbeta_off = np.exp(xbeta + offset + cnv)
        frac = (counts + size) * size * exp_xbeta_off / (size + exp_xbeta_off) ** 2
        # Build diagonal
        h11 = 1 / prior_no_shrink_scale**2
        h22 = (
            2
            * (prior_scale**2 - beta[shrink_index] ** 2)
            / (prior_scale**2 + beta[shrink_index] ** 2) ** 2
        )

        h = np.diag(no_shrink_mask * h11 + shrink_mask * h22)

        return 1 / cnst * ((design_matrix.T * frac) @ design_matrix + np.diag(h))

    res = minimize(
        f,
        beta_init,
        jac=df,
        hess=ddf if optimizer == "Newton-CG" else None,
        method=optimizer,
    )

    beta = res.x
    converged = res.success

    if not converged and num_vars == 2:
        # If the solver failed, fit using grid search (slow)
        # Only for single-factor analysis
        beta = grid_fit_shrink_beta(
            counts,
            cnv,
            offset,
            design_matrix,
            size,
            prior_no_shrink_scale,
            prior_scale,
            scale_cnst,
            grid_length=60,
            min_beta=-30,
            max_beta=30,
        )

    inv_hessian = np.linalg.inv(ddf(beta, 1))

    return beta, inv_hessian, converged
    

def nbinomFn(
    beta: np.ndarray,
    design_matrix: np.ndarray,
    counts: np.ndarray,
    cnv: np.ndarray,
    size: np.ndarray,
    offset: np.ndarray,
    prior_no_shrink_scale: float,
    prior_scale: float,
    shrink_index: int = 1,
) -> float:
    """Return the NB negative likelihood with apeGLM prior.

    Use for LFC shrinkage.

    Parameters
    ----------
    beta : ndarray
        2-element array: intercept and LFC coefficients.

    design_matrix : ndarray
        Design matrix.

    counts : ndarray
        Raw counts.

    size : ndarray
        Size parameter of NB family (inverse of dispersion).

    offset : ndarray
        Natural logarithm of size factor.

    prior_no_shrink_scale : float
        Prior variance for the intercept.

    prior_scale : float
        Prior variance for the intercept.

    shrink_index : int
        Index of the LFC coordinate to shrink. (default: ``1``).

    Returns
    -------
    float
        Sum of the NB negative likelihood and apeGLM prior.
    """
    num_vars = design_matrix.shape[-1]

    shrink_mask = np.zeros(num_vars)
    shrink_mask[shrink_index] = 1
    no_shrink_mask = np.ones(num_vars) - shrink_mask

    xbeta = design_matrix @ beta
    prior = (
        (beta * no_shrink_mask) ** 2 / (2 * prior_no_shrink_scale**2)
    ).sum() + np.log1p((beta[shrink_index] / prior_scale) ** 2)

    nll = (
        counts * xbeta - (counts + size) * np.logaddexp(xbeta + offset + cnv, np.log(size))
    ).sum(0)

    return prior - nll


def run_stageR(
    res_pydeseq,
    res_deconveil,
    screen_col="pvalue",
    confirm_col="pvalue",
    alpha=0.05,
    method="holm",
):
    """
    Two-stage gene-level multiple testing using stageR.

    Stage I (screening):
        - Omnibus Simes test combining CN-naive and CN-aware p-values
        - BH FDR applied once across genes

    Stage II (confirmation):
        - Within-gene multiplicity correction (Holm) on naive + aware tests
        - Conditional on passing Stage I

    Parameters
    ----------
    res_pydeseq : pd.DataFrame
        CN-naive DE results with raw p-values
    res_deconveil : pd.DataFrame
        CN-aware DE results with raw p-values
    screen_col : str
        Column name of raw p-values (used for screening)
    confirm_col : str
        Column name of raw p-values (used for confirmation)
    alpha : float
        Target FDR level
    method : str
        Within-gene correction method (e.g. "holm")

    Returns
    -------
    res_screen : pd.DataFrame
        Adjusted screening p-values (gene-level)
    res_confirm : pd.DataFrame
        0/1 confirmation decisions per hypothesis
    res_naive_upd : pd.DataFrame
        CN-naive results with stageR-adjusted q-values
    res_aware_upd : pd.DataFrame
        CN-aware results with stageR-adjusted q-values
    """

    # --------------------------------------------------
    # 1. Extract raw p-values
    # --------------------------------------------------
    p_naive = res_pydeseq[screen_col].astype(float)
    p_aware = res_deconveil[screen_col].astype(float)

    # Ensure alignment
    p_naive, p_aware = p_naive.align(p_aware, join="inner")

    # --------------------------------------------------
    # 2. Omnibus screening p-values (Simes)
    # --------------------------------------------------
    p1 = np.minimum(p_naive, p_aware)
    p2 = np.maximum(p_naive, p_aware)
    p_screen = np.minimum(2.0 * p1, p2)

    #p_screen = pd.Series(p_screen, index=p_naive.index, name="p_screen")

    # --------------------------------------------------
    # 3. Confirmation p-values matrix
    # --------------------------------------------------
    
    p_naive_conf = pd.DataFrame({"p_naive": res_pydeseq[confirm_col].astype(float)})
    p_aware_conf = pd.DataFrame({"p_aware": res_deconveil[confirm_col].astype(float)})
    p_conf = pd.concat([p_naive_conf, p_aware_conf], axis=1)

    # stageR requires string rownames
    p_screen.index = p_screen.index.astype(str)
    p_conf.index = p_conf.index.astype(str)

    # --------------------------------------------------
    # 4. Convert to R
    # --------------------------------------------------

    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        r_p_screen = conversion.py2rpy(p_screen)
        r_p_conf   = conversion.py2rpy(p_conf)

    # Assign R variables
    genes = list(p_conf.index) 
    ro.globalenv["p_screen"] = r_p_screen
    ro.globalenv["p_conf"] = r_p_conf
    ro.globalenv["genes"] = ro.StrVector(list(genes))
    ro.globalenv["conf_names"] = ro.StrVector(list(p_conf.columns))

    # --------------------------------------------------
    # 5. Run stageR
    # --------------------------------------------------
    r_code = f"""
        library(stageR)

        p_conf <- as.matrix(p_conf)

        stageRObj <- stageR(
            pScreen = p_screen,
            pConfirmation = p_conf,
            pScreenAdjusted = FALSE
        )

        stageRObj <- stageWiseAdjustment(
            stageRObj,
            method = "{method}",
            alpha = {alpha},
            #allowNA = TRUE
        )

        res_screen <- getAdjustedPValues(
            stageRObj,
            onlySignificantGenes = FALSE,
            order = FALSE
        )

        res_confirm <- getResults(stageRObj)
    """

    ro.r(r_code)

    # --------------------------------------------------
    # 6. Convert back to Python
    # --------------------------------------------------
    
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        res_screen = conversion.rpy2py(ro.r("res_screen"))
        res_confirm = conversion.rpy2py(ro.r("res_confirm"))

    # Ensure pandas DataFrames
    if isinstance(res_screen, np.ndarray):
        rows = list(ro.r("rownames(res_screen)"))
        cols = list(ro.r("colnames(res_screen)"))
        res_screen = pd.DataFrame(res_screen, index=rows, columns=cols)

    if isinstance(res_confirm, np.ndarray):
        rows = list(ro.r("rownames(res_confirm)"))
        cols = list(ro.r("colnames(res_confirm)"))
        res_confirm = pd.DataFrame(res_confirm, index=rows, columns=cols)

    # --------------------------------------------------
    # 7. Attach results to original tables
    # --------------------------------------------------

    res_screen.index = res_screen.index.astype(str)

    # 1) Update PyDESeq2 table with SCREEN q-values
    res_pydeseq_upd = res_pydeseq.copy()
    if "p_naive" in res_screen.columns:
        res_pydeseq_upd["padj_stageR"] = (
            res_screen["p_naive"].reindex(res_pydeseq_upd.index.astype(str)).values
        )

    # 2) Update DeConveil table with SCREEN q-values
    res_deconveil_upd = res_deconveil.copy()
    if "p_aware" in res_screen.columns:
        res_deconveil_upd["padj_stageR"] = (
            res_screen["p_aware"].reindex(res_pydeseq_upd.index.astype(str)).values
        )

    res_confirm.index = res_confirm.index.astype(str)
    if "p_naive" in res_confirm.columns:
        res_pydeseq_upd["DE_confirmed"] = (
            res_confirm["p_naive"].reindex(res_pydeseq_upd.index.astype(str)).values
        )

    if "p_aware" in res_confirm.columns:
        res_deconveil_upd["DE_confirmed"] = (
            res_confirm["p_aware"].reindex(res_deconveil_upd.index.astype(str)).values
        )
    
    # NA = not tested / not confirmed
  
    return res_screen, res_confirm, res_pydeseq_upd, res_deconveil_upd