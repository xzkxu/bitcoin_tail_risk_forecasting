import numpy as np
import torch
import pandas as pd
from scipy.stats import chi2
from scipy.stats import genpareto

# === LSTM loss === 

def fz0_loss_train(y, v, e, alpha, kappa=20.0, eps=1e-6):
    I = torch.sigmoid(kappa * (v - y))  # Smooth indicator ~ 1{y ≤ v}
    e_safe = torch.clamp(e, max=-eps)   # e must be strictly negative
    term1 = (I * (v - y)) / (alpha * (-e_safe))
    term2 = v / e_safe
    term3 = torch.log(-e_safe)
    L = term1 + term2 + term3 - 1.0
    return L.mean()

def pinball_loss(y, v, alpha):
    u = y - v
    return (alpha * torch.clamp(u, min=0.0) + (1 - alpha) * torch.clamp(-u, min=0.0)).mean()


# === VaR Backtests ===
def kupiec_pof_test(y, q, alpha):
    """Kupiec unconditional coverage test"""
    n = len(y)
    hits = (y < q).sum()
    pi_hat = hits / n
    if hits == 0 or hits == n:
        return np.nan, np.nan
    LR = -2 * (np.log((1 - alpha)**(n-hits) * alpha**hits) -
               np.log((1 - pi_hat)**(n-hits) * (pi_hat**hits)))
    pval = 1 - chi2.cdf(LR, df=1)
    return LR, pval

def christoffersen_independence_test(y, q):
    """Christoffersen independence test for violation clustering"""
    hits = (y < q).astype(int)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(hits)):
        if hits[i-1] == 0 and hits[i] == 0: n00 += 1
        elif hits[i-1] == 0 and hits[i] == 1: n01 += 1
        elif hits[i-1] == 1 and hits[i] == 0: n10 += 1
        elif hits[i-1] == 1 and hits[i] == 1: n11 += 1
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi_hat = (n01 + n11) / (n00 + n01 + n10 + n11)
    LR_ind = -2 * (
        np.log((1 - pi_hat)**(n00+n10) * (pi_hat**(n01+n11))) -
        np.log((1 - pi01)**n00 * (pi01**n01) *
               (1 - pi11)**n10 * (pi11**n11))
    )
    pval = 1 - chi2.cdf(LR_ind, df=1)
    return LR_ind, pval

def christoffersen_cc_test(y, q, alpha):
    """Conditional coverage = POF + independence"""
    LR_pof, _ = kupiec_pof_test(y, q, alpha)
    LR_ind, _ = christoffersen_independence_test(y, q)
    LR_cc = LR_pof + LR_ind
    pval = 1 - chi2.cdf(LR_cc, df=2)
    return LR_cc, pval

# === ES Backtest (Acerbi–Szekely) ===
def acerbi_szekely_test(y, q, es, alpha):
    """Simple ES backtest statistic"""
    hits = (y < q).astype(int)
    if hits.sum() == 0:
        return np.nan, np.nan
    S = hits * (y - es) / alpha
    t_stat = S.mean() / (S.std(ddof=1) / np.sqrt(len(S)))
    from scipy.stats import t
    pval = 2 * (1 - t.cdf(abs(t_stat), df=len(S)-1))
    return t_stat, pval

# === Evaluation helper ===

def fz_loss(y, q, es, alpha):
    y = np.asarray(y); q = np.asarray(q); es = np.asarray(es)
    I = (y < q).astype(int)
    valid = np.isfinite(es) & (es != 0)
    if valid.sum() == 0:
        return np.nan
    I, y, q, es = I[valid], y[valid], q[valid], es[valid]
    return np.mean(((I - alpha) * (q - y) / alpha) + I * (((q - y) / es) - 1) / alpha)

def pot_var_es(std_resid, sigma_t1, thresh_q, target_alphas):
    Z = np.asarray(std_resid.dropna())
    N = len(Z)
    if N < 100:
        return {}
    Y = -Z
    y_u = np.quantile(Y, 1.0 - thresh_q)
    E = Y[Y > y_u] - y_u
    N_u = E.size
    p_u = N_u / N
    if N_u < 20 or p_u <= 0:
        return {}
    try:
        xi, loc, beta = genpareto.fit(E, floc=0.0)
    except:
        return {}
    if not np.isfinite(xi) or not np.isfinite(beta) or beta <= 0 or xi >= 1:
        return {}
    sd_Z = Z.std()
    if sd_Z <= 0 or not np.isfinite(sd_Z):
        return {}
    out = {}
    for alpha in target_alphas:
        r = alpha / p_u
        if r <= 0 or r >= 1:
            continue
        VaR_Y = y_u + (beta / xi) * (r**(-xi) - 1.0)
        ES_Y  = (VaR_Y + beta - xi * y_u) / (1.0 - xi)
        VaR_perc = (-VaR_Y) * (sigma_t1 / sd_Z)
        ES_perc  = (-ES_Y)  * (sigma_t1 / sd_Z)
        if np.isfinite(VaR_perc) and np.isfinite(ES_perc):
            out[alpha] = (float(VaR_perc), float(ES_perc))
    return out

def evaluate_alpha(df_alpha, alpha):
    y, q, es = df_alpha['Realized'], df_alpha['VaR'], df_alpha['ES']
    qloss = np.mean((alpha - (y < q).astype(int)) * (y - q))
    fz = fz_loss(y, q, es, alpha)
    hit_ratio = (y < q).sum() / (alpha * len(y))
    es_err = y[y < q].mean() - es[y < q].mean() if (y < q).sum() > 0 else np.nan
    return {'QuantileLoss': qloss, 'FZLoss': fz, 'HitRatio': hit_ratio, 'ESError': es_err}

def quantile_loss(y, v, alpha):
    """
    Pinball/quantile loss for VaR at level alpha.
    y, v: 1D numpy arrays (realized, VaR).
    Returns the sample MEAN loss.
    """
    y = np.asarray(y); v = np.asarray(v)
    u = y - v
    return np.mean(alpha * np.maximum(u, 0.0) + (1.0 - alpha) * np.maximum(-u, 0.0))

def evaluate_results_same_as_garch(results_df, alphas):
    """
    results_df: MultiIndex DataFrame indexed by ['Date','Alpha'] with columns ['VaR','ES','Realized']
    alphas: list of alpha levels (e.g., [0.05, 0.025, 0.01])
    es_is_negative: True if ES column is negative (typical). If your model stores -ES (>0),
                    set es_is_negative=False to flip before scoring.

    Returns: DataFrame with N, HitRate, QuantileLoss, FZLoss for each alpha.
    """
    rows = []
    for a in alphas:
        df_a = results_df.xs(a, level='Alpha')
        y = df_a['Realized'].to_numpy()
        v = df_a['VaR'].to_numpy()
        e = df_a['ES'].to_numpy()

        ql = quantile_loss(y, v, a)
        fz = fz_loss(y, v, e, a)
        hit_rate = float(np.mean(y <= v))  # keep <= here if you prefer, otherwise use (y < v)
        rows.append({'Alpha': a, 'N': len(y), 'HitRate': hit_rate,
                     'QuantileLoss': ql, 'FZLoss': fz})
    return pd.DataFrame(rows).set_index('Alpha').sort_index()