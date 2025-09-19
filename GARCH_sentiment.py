import json
import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import genpareto
import matplotlib.pyplot as plt
from functions import *

# --- Config ---
TRAIN_START = '2022-1-20'
TRAIN_END   = '2023-1-19'
TEST_START  = '2023-1-20'
TEST_END    = '2023-12-19'
WINDOW = 360
THRESH_Q = 0.2
ALPHAS = [0.05, 0.025]  # focusing on 2.5% tail

def main():
    # ========= 1) Load sentiment JSON =========
    with open("bitcoin_sentiment_llm.json", "r") as f:
        raw_sent = json.load(f)

    # ========= 2) Map to numeric scores =========
    label_map = {"Bullish": 1, "Neutral": 0, "Bearish": -1}

    rows = []
    for date_str, daily_dict in raw_sent.items():
        # Convert inner dict to list of sentiment strings
        labels = list(daily_dict.values())
        scores = [label_map.get(lbl, 0) for lbl in labels]

        N_t = len(scores)  # number of headlines
        if N_t == 0:
            mean_score = 0
        else:
            mean_score = np.mean(scores)  # polarity in [-1,1]

        # Volatility-relevant sentiment factor: absolute polarity × log volume
        A_t = abs(mean_score) * np.log(1 + N_t)

        rows.append({"date": pd.to_datetime(date_str), 
                    "S_t": mean_score,  # signed polarity
                    "N_t": N_t,         # number of headlines
                    "A_t": A_t})        # volatility-relevant factor

    df_sent = pd.DataFrame(rows).sort_values("date").set_index("date")

    # ========= 3) Lag to avoid look-ahead =========
    df_sent["A_lag"] = df_sent["A_t"].shift(1).fillna(0)

    # ========= 4) Standardize on training set only =========
    train_mask = (df_sent.index >= TRAIN_START) & (df_sent.index <= TRAIN_END)
    mu_A = df_sent.loc[train_mask, "A_lag"].mean()
    sigma_A = df_sent.loc[train_mask, "A_lag"].std(ddof=0)

    df_sent["A_std"] = (df_sent["A_lag"] - mu_A) / (sigma_A if sigma_A > 0 else 1)



    df = pd.read_csv('btc_usd.csv', parse_dates=['date'], index_col='date').sort_index()

    # Merge
    df = df.merge(df_sent[['A_std']], left_index=True, right_index=True, how='left')

    # --- Create returns & features ---
    df['r'] = np.log(df['close']).diff() * 100  # % returns
    df['dlog_vol'] = np.log(df['volume'] + 1).diff()

    # Lag features
    df['r_lag1'] = df['r'].shift(1)
    df['A_lag'] = df['A_std'].shift(1)

    # Drop missing
    data = df[['r', 'dlog_vol', 'r_lag1', 'A_lag']].dropna()



    # --- Rolling forecast ---
    records = []
    test_idx = data.loc[TEST_START:TEST_END].index
    for test_date in test_idx:
        i = data.index.get_loc(test_date)
        train_slice = data.iloc[i-WINDOW:i]

        y_train = train_slice['r']
        X_train = train_slice[['dlog_vol', 'A_lag']]  # Include sentiment factor

        am = arch_model(y_train, mean='ARX', lags=1, x=X_train, vol='GARCH', p=1, q=1, dist='t')
        res = am.fit(disp='off')

        std_resid = res.std_resid.dropna()
        # std_resid = res.std_resid.dropna()  # keep as-is

        x_forecast = {
            'dlog_vol': np.array([data.at[test_date, 'dlog_vol']]),
            'A_lag':    np.array([data.at[test_date, 'A_lag']])
        }
        sigma_t1 = res.forecast(horizon=1, x=x_forecast, reindex=False).variance.values[-1, 0] ** 0.5

        for alpha in ALPHAS:
            out = pot_var_es(std_resid, sigma_t1, THRESH_Q, alpha)
            if out:
                var_a, es_a = out
                records.append({
                    'Date': test_date,
                    'Alpha': alpha,
                    'VaR': var_a,
                    'ES': es_a,
                    'Realized': data.loc[test_date, 'r']
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
