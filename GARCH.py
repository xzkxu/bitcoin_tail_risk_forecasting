import pandas as pd
import numpy as np
from arch import arch_model
from functions import *

# ===== Date Config =====
# TRAIN_START = '2015-07-21'
# TRAIN_END   = '2020-07-20'
# TEST_START  = '2020-07-21'
# TEST_END    = '2025-07-20'
TRAIN_START = '2022-1-20'
TRAIN_END   = '2023-1-19'
TEST_START  = '2023-1-20'
TEST_END    = '2023-12-19'

# ===== Model Config =====
# WINDOW = 1800         
# THRESH_Q = 0.25      
# ALPHAS = [0.05, 0.025, 0.01]
WINDOW = 360         # days for rolling window
THRESH_Q = 0.8      # EVT threshold quantile
ALPHAS = [0.05, 0.025]

def main():
        # Load
    df = pd.read_csv('btc_usd.csv', parse_dates=['date'], index_col='date').sort_index()

    # Features
    df['r'] = np.log(df['close']).diff()
    eps = 1e-12
    lnH_L = np.log((df['high'] + eps) / (df['low'] + eps))
    df['rv_parkinson'] = (lnH_L ** 2) / (4.0 * np.log(2.0))
    df['log_vol']  = np.log(df['volume'] + 1.0)
    df['dlog_vol'] = df['log_vol'].diff()
    df['r_lag1'] = df['r'].shift(1)

    # Drop NaNs
    data = df[['r','dlog_vol','r_lag1']].dropna()

    # Select test dates
    test_dates = data.loc[TEST_START:TEST_END].index

    records = []
    for test_date in test_dates:
        # Window for training ends at t-1
        train_end_date = data.index[data.index.get_loc(test_date) - 1]
        train_start_idx = max(0, data.index.get_loc(train_end_date) - WINDOW + 1)
        train_data = data.iloc[train_start_idx : data.index.get_loc(train_end_date) + 1]
        
        y_train = train_data['r'] * 100
        X_train = train_data[['dlog_vol']]

        # Fit GARCH-X
        try:
            am = arch_model(y_train, mean='ARX', lags=1, x=X_train,
                            vol='GARCH', p=1, q=1, dist='t')
            res = am.fit(disp='off', options={'maxiter': 2000})
        except:
            continue

        std_resid = res.std_resid.dropna()
        sigma_t1 = res.forecast(horizon=1, x=data.loc[[test_date], ['dlog_vol']].values).variance.values[-1,0] ** 0.5

        # EVT
        alpha2qe = pot_var_es(std_resid, sigma_t1, THRESH_Q, ALPHAS)
        for a in ALPHAS:
            if a in alpha2qe:
                q_a, es_a = alpha2qe[a]
                records.append({
                    'Date': test_date,
                    'Alpha': a,
                    'VaR': q_a,
                    'ES': es_a,
                    'Realized': float(data.loc[test_date, 'r'] * 100)
                })

    results = pd.DataFrame(records).set_index(['Date','Alpha']).sort_index()

    metrics = {a: evaluate_alpha(results.xs(a, level='Alpha'), a) for a in ALPHAS}
    metrics_df = pd.DataFrame(metrics).T
    print("GARCH metrics:\n", metrics_df)

    for a in ALPHAS:
        if (results.index.get_level_values('Alpha') == a).any():
            df_a = results.xs(a, level='Alpha').dropna(subset=['VaR','Realized'])
            y = df_a['Realized'].values
            q = df_a['VaR'].values
            es = df_a['ES'].values

            LR_pof, p_pof = kupiec_pof_test(y, q, a)
            LR_ind, p_ind = christoffersen_independence_test(y, q)
            LR_cc, p_cc = christoffersen_cc_test(y, q, a)
            t_as, p_as = acerbi_szekely_test(y, q, es, a)

            print(f"\nAlpha={a:.3%} — Backtests")
            print(f"Kupiec POF       : LR={LR_pof:.4f}, p={p_pof:.4f}")
            print(f"Christ. Indep.   : LR={LR_ind:.4f}, p={p_ind:.4f}")
            print(f"Christ. CondCov  : LR={LR_cc:.4f}, p={p_cc:.4f}")
            print(f"Acerbi–Szekely ES: t={t_as:.4f}, p={p_as:.4f}")


if __name__ == "__main__":
    main()