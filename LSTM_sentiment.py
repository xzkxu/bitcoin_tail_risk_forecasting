import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
from functions import *

class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).view(-1, 1)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

class VaRESLSTM(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=1, dropout=0.1, margin=1e-3):
        super().__init__()
        self.margin = margin
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=0.0 if layers==1 else dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        z = self.head(h)
        z_v, z_gap = z[:, :1], z[:, 1:2]
        v = -F.softplus(z_v) - self.margin       # VaR < 0
        e = v - F.softplus(z_gap) - self.margin  # ES < VaR
        return v, e

TRAIN_START = '2022-01-20'
TRAIN_END   = '2023-01-19'
TEST_START  = '2023-01-20'
TEST_END    = '2023-12-19'
ALPHAS = [0.05, 0.025]

ALPHA_CONFIGS = {
    # 0.01:  dict(SEQ_LEN=90,  HIDDEN_SIZE=96,  SMOOTH_KAPPA=12, UPDATE_EVERY=3,
    #             CALIB_WINDOW=180, MIN_CALIB_N=120, GRID_STEPS=80, LAMBDA_MAX=3.0,
    #             WARMUP_EPOCHS=12, CALIB_EPOCHS=4),
    0.025: dict(SEQ_LEN=90,  HIDDEN_SIZE=64,  SMOOTH_KAPPA=18, UPDATE_EVERY=5,
                CALIB_WINDOW=200, MIN_CALIB_N=150,  GRID_STEPS=120, LAMBDA_MAX=3.0,
                WARMUP_EPOCHS=12, CALIB_EPOCHS=4),
    0.05:  dict(SEQ_LEN=60,  HIDDEN_SIZE=64,  SMOOTH_KAPPA=25, UPDATE_EVERY=7,
                CALIB_WINDOW=90,  MIN_CALIB_N=60,  GRID_STEPS=50, LAMBDA_MAX=1.5,
                WARMUP_EPOCHS=8,  CALIB_EPOCHS=3),
}

BATCH_SIZE     = 128
NUM_LAYERS     = 1
DROPOUT        = 0.10
LR_PRETRAIN    = 5e-4
LR_FINETUNE    = 3e-4
EPOCHS_PRE     = 50
PATIENCE       = 6
MARGIN         = 1e-3
RNG_SEED       = 123
DEVICE         = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(4)
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)

def main():
    df = pd.read_csv('btc_usd.csv', parse_dates=['date'], index_col='date').sort_index()

    eps = 1e-12
    df['r'] = np.log(df['close']).diff()
    df['r_pct'] = df['r'] * 100.0

    lnH_L = np.log((df['high'] + eps) / (df['low'] + eps))
    df['rv_parkinson'] = (lnH_L ** 2) / (4.0 * np.log(2.0))
    df['log_vol']  = np.log(df['volume'] + 1.0)
    df['dlog_vol'] = df['log_vol'].diff()

    df['r_lag1'] = df['r_pct'].shift(1)
    df['r_lag2'] = df['r_pct'].shift(2)
    df['abs_r']  = df['r_pct'].abs()
    df['r2']     = df['r_pct']**2
    df['rv_roll_14'] = df['r_pct'].rolling(14).std()
    df['rv_roll_30'] = df['r_pct'].rolling(30).std()

    # ----------------
    # NEW: Load & engineer sentiment features (no leakage)
    # bitcoin_sentiment_llm.json structure:
    # {
    #   "YYYY-MM-DD": {"0":"Bullish","1":"Neutral","2":"Bearish",...},
    #   ...
    # }
    # We map Bullish=+1, Bearish=-1, Neutral=0 and form daily aggregates.
    # ----------------
    with open('bitcoin_sentiment_llm.json', 'r') as f:
        raw_sent = json.load(f)

    rows = []
    for d_str, labels_dict in raw_sent.items():
        pos = neg = neu = 0
        for lab in labels_dict.values():
            s = str(lab).strip().lower()
            if s.startswith('bull'):  pos += 1
            elif s.startswith('bear'): neg += 1
            else:                      neu += 1
        total = pos + neg + neu
        score = (pos - neg) / total if total > 0 else 0.0
        rows.append({
            'date': pd.to_datetime(d_str),
            'sent_pos': pos, 'sent_neg': neg, 'sent_neu': neu,
            'sent_total': total, 'sent_score': score
        })

    sent = sent.reindex(df.index)
    for c in ['sent_pos','sent_neg','sent_neu','sent_total']:
        sent[c] = sent[c].fillna(0)
    sent['sent_score'] = sent['sent_score'].fillna(0.0)

    # Smoothed/scale features (computed on same-day value; your sequence uses up to t-1, so no leakage)
    sent['log_sent_count'] = np.log(sent['sent_total'] + 1.0)
    sent['sent_ewm_7']  = sent['sent_score'].ewm(span=7,  adjust=False).mean()
    sent['sent_ewm_30'] = sent['sent_score'].ewm(span=30, adjust=False).mean()

    # Merge back to main frame
    df = df.join(sent[['sent_score','sent_ewm_7','sent_ewm_30','log_sent_count']], how='left')

    # ----------------
    # Features list (extended with sentiment)
    # ----------------
    FEATS = [
        'r_lag1','r_lag2','dlog_vol','rv_parkinson','abs_r','r2',
        'rv_roll_14','rv_roll_30',
        # --- NEW sentiment features ---
        'sent_score','sent_ewm_7','sent_ewm_30','log_sent_count'
    ]

    base = df[['r_pct'] + FEATS].dropna().copy()

    # ----------------
    # Helpers to build sequences & masks per α (since SEQ_LEN differs)
    # ----------------
    def make_sequences(frame, target_col, feature_cols, seq_len):
        X_list, y_list, dates = [], [], []
        vals = frame[feature_cols].values.astype(np.float32)
        tgt  = frame[target_col].values.astype(np.float32)
        idx  = frame.index
        for t in range(seq_len, len(frame)):
            # Uses features up to t-1 (no leakage) to predict y_t
            X_list.append(vals[t-seq_len:t])
            y_list.append(tgt[t])
            dates.append(idx[t])
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        dates = pd.to_datetime(np.array(dates))
        return X, y, dates

    def build_masks(dates):
        TRAIN_START_TS = pd.Timestamp(TRAIN_START)
        TRAIN_END_TS   = pd.Timestamp(TRAIN_END)
        TEST_START_TS  = pd.Timestamp(TEST_START)
        TEST_END_TS    = pd.Timestamp(TEST_END)
        train_mask = (dates >= TRAIN_START_TS) & (dates <= TRAIN_END_TS)
        test_mask  = (dates >= TEST_START_TS)  & (dates <= TEST_END_TS)
        train_dates = dates[train_mask]
        if len(train_dates) < 50:
            raise RuntimeError("Training set too small after sequence construction.")
        cut_idx = int(len(train_dates) * 0.80)
        val_cut_date = train_dates[cut_idx]
        pretrain_mask = train_mask & (dates <= val_cut_date)
        val_mask      = train_mask & (dates >  val_cut_date)
        return pretrain_mask, val_mask, test_mask
    
    # ----------------
    # Calibration (shift-only, α-targeted)
    # ----------------
    def calibrate_var_es_shift(v_raw_t, e_raw_t, alpha, hist, cfg):
        """
        hist: list of dicts {'y', 'v_raw'} from strictly past days (t-1, t-2, ...)
        cfg:  dict with CALIB_WINDOW, MIN_CALIB_N, GRID_STEPS, LAMBDA_MAX
        """
        CALIB_WINDOW = cfg['CALIB_WINDOW']
        MIN_CALIB_N  = cfg['MIN_CALIB_N']
        GRID_STEPS   = cfg['GRID_STEPS']
        LAMBDA_MAX   = cfg['LAMBDA_MAX']

        if CALIB_WINDOW is not None and len(hist) > CALIB_WINDOW:
            hist = hist[-CALIB_WINDOW:]
        if len(hist) < MIN_CALIB_N:
            v_cal = min(v_raw_t, -1e-3)
            e_cal = min(e_raw_t, v_cal - 1e-3)
            return float(v_cal), float(e_cal)

        y_hist = np.array([h['y'] for h in hist], dtype=float)
        v_hist = np.array([h['v_raw'] for h in hist], dtype=float)

        resid = y_hist - v_hist
        c_hat = np.quantile(resid, alpha)

        # λ grid search for closest coverage to α
        lam_grid = np.linspace(0.0, LAMBDA_MAX, GRID_STEPS)
        best_lam, best_gap, best_tail_mean = 0.0, 1e9, None
        for lam in lam_grid:
            thr_hist = np.minimum(0.0, v_hist + lam * c_hat)
            cov = float(np.mean(y_hist <= thr_hist))
            gap = abs(cov - alpha)
            if gap < best_gap:
                best_gap = gap
                best_lam = lam
                tail_mask = (y_hist <= thr_hist)
                best_tail_mean = y_hist[tail_mask].mean() if tail_mask.sum() >= 5 else None

        v_cal = min(v_raw_t + best_lam * c_hat, -1e-3)
        e_cal = best_tail_mean if best_tail_mean is not None else e_raw_t
        e_cal = min(e_cal, v_cal - 1e-3)
        return float(v_cal), float(e_cal)

    def calibrate_var_es_exact(v_raw_t, e_raw_t, alpha, hist, cfg):
        """
        Exact coverage calibration (shift-only) on a rolling past window.
        We find the smallest λ ≥ 0 such that coverage(y_hist <= min(0, v_hist + λ * c_hat)) ≈ α,
        where c_hat is the empirical α-quantile of residuals (y_hist - v_hist).
        ES is recomputed as the tail mean under the calibrated thresholds on the same window.
        """
        CALIB_WINDOW = cfg['CALIB_WINDOW']
        MIN_CALIB_N  = cfg['MIN_CALIB_N']
        LAMBDA_MAX   = cfg['LAMBDA_MAX']

        if CALIB_WINDOW is not None and len(hist) > CALIB_WINDOW:
            hist = hist[-CALIB_WINDOW:]
        if len(hist) < MIN_CALIB_N:
            v_cal = min(v_raw_t, -1e-3)
            e_cal = min(e_raw_t, v_cal - 1e-3)
            return float(v_cal), float(e_cal)

        y_hist = np.array([h['y']     for h in hist], dtype=float)
        v_hist = np.array([h['v_raw'] for h in hist], dtype=float)

        resid = y_hist - v_hist
        c_hat = np.quantile(resid, alpha)
        if c_hat >= -1e-8:
            fallback_q = max(alpha/2.0, 1.0/len(resid))
            c_hat = min(-1e-4, np.quantile(resid, fallback_q))

        def coverage(lmb):
            thr = np.minimum(0.0, v_hist + lmb * c_hat)
            return float(np.mean(y_hist <= thr)), thr

        target = alpha
        tol = max(1.0/len(y_hist), 0.001)

        cov0, _ = coverage(0.0)
        if cov0 <= target + tol:
            lam_star = 0.0
            thr_hist = np.minimum(0.0, v_hist + lam_star * c_hat)
        else:
            lam_hi = LAMBDA_MAX
            cov_hi, thr_hi = coverage(lam_hi)
            attempts = 0
            while cov_hi > target + tol and attempts < 12:
                lam_hi *= 1.5
                cov_hi, thr_hi = coverage(lam_hi)
                attempts += 1
            lo, hi = 0.0, lam_hi
            for _ in range(32):
                mid = 0.5 * (lo + hi)
                cov_mid, _ = coverage(mid)
                if cov_mid > target:
                    lo = mid
                else:
                    hi = mid
            lam_star = hi
            _, thr_hist = coverage(lam_star)

        v_cal = min(v_raw_t + lam_star * c_hat, -1e-3)

        tail_mask = (y_hist <= thr_hist)
        if tail_mask.sum() >= 5:
            e_cal = float(y_hist[tail_mask].mean())
        else:
            e_cal = e_raw_t
        e_cal = min(e_cal, v_cal - 1e-3)
        return float(v_cal), float(e_cal)

    # ----------------
    # One α end-to-end run
    # ----------------
    def run_for_alpha(alpha, cfg):
        # 1) Build sequences for this α (its SEQ_LEN)
        X_all, y_all, dates_all = make_sequences(base, 'r_pct', FEATS, cfg['SEQ_LEN'])
        pretrain_mask, val_mask, test_mask = build_masks(dates_all)

        # 2) Fit scaler on pretraining window only (no leakage)
        scaler = StandardScaler().fit(X_all[pretrain_mask].reshape(-1, X_all.shape[-1]))
        X_all_scaled = scaler.transform(X_all.reshape(-1, X_all.shape[-1])).reshape(X_all.shape)

        # 3) Datasets
        ds_pre  = SeqDS(X_all_scaled[pretrain_mask], y_all[pretrain_mask])
        ds_val  = SeqDS(X_all_scaled[val_mask],      y_all[val_mask])

        # 4) Model and training
        model = VaRESLSTM(in_dim=len(FEATS), hidden=cfg['HIDDEN_SIZE'],
                        layers=NUM_LAYERS, dropout=DROPOUT, margin=MARGIN).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR_PRETRAIN, weight_decay=1e-6)
        dl_pre = DataLoader(ds_pre, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

        best_val, best_state, no_improve = float('inf'), None, 0

        # ---- Warm-start with pinball on VaR only ----
        for _ in range(cfg['WARMUP_EPOCHS']):
            model.train()
            for Xb, yb in dl_pre:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                v, e = model(Xb)
                loss = pinball_loss(yb, v, alpha)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        # ---- Joint FZ0 training with early stopping ----
        for epoch in range(EPOCHS_PRE):
            model.train()
            for Xb, yb in dl_pre:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                v, e = model(Xb)
                loss = fz0_loss_train(yb, v, e, alpha, kappa=cfg['SMOOTH_KAPPA'])
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            # validation
            model.eval()
            with torch.no_grad():
                val_loss, n = 0.0, 0
                for Xb, yb in dl_val:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    v, e = model(Xb)
                    b = Xb.size(0)
                    val_loss += fz0_loss_train(yb, v, e, alpha, kappa=cfg['SMOOTH_KAPPA']).item() * b
                    n += b
                val_loss /= max(1, n)

            if val_loss < best_val - 1e-5:
                best_val, best_state, no_improve = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # 5) Fine-tuning helper (data up to cutoff_date)
        def finetune_to_date(cutoff_date):
            mask = dates_all <= cutoff_date
            ds = SeqDS(X_all_scaled[mask], y_all[mask])
            if len(ds) == 0:
                return
            dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
            opt_ft = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-6)
            model.train()
            for _ in range(cfg['CALIB_EPOCHS']):
                for Xb, yb in dl:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    v, e = model(Xb)
                    loss = fz0_loss_train(yb, v, e, alpha, kappa=cfg['SMOOTH_KAPPA'])
                    opt_ft.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt_ft.step()
            model.eval()

        # 6) Walk-forward forecast with per-α update cadence & calibration
        test_idx = np.where(test_mask)[0]
        records, hist = [], []
        last_update_date = None

        for i in test_idx:
            d = pd.Timestamp(dates_all[i]).normalize()
            if (last_update_date is None) or ((d - last_update_date).days >= cfg['UPDATE_EVERY']):
                cutoff = d - pd.Timedelta(days=1)
                finetune_to_date(cutoff)
                last_update_date = d

            with torch.no_grad():
                Xb = torch.from_numpy(X_all_scaled[i:i+1]).to(DEVICE)
                v_raw, e_raw = model(Xb)
                v_raw = float(v_raw.item()); e_raw = float(e_raw.item())

            # Calibrate with α-specific window & strength
            if cfg.get('CALIBRATOR', 'shift') == 'exact':
                v_cal, e_cal = calibrate_var_es_exact(v_raw, e_raw, alpha, hist, cfg)
            else:
                v_cal, e_cal = calibrate_var_es_shift(v_raw, e_raw, alpha, hist, cfg)

            realized = float(y_all[i])
            records.append({'Date': d, 'Alpha': alpha, 'VaR': v_cal, 'ES': e_cal, 'Realized': realized})

            # update history with today's realized and RAW VaR (no leakage)
            hist.append({'y': realized, 'v_raw': v_raw})

        res = pd.DataFrame(records).set_index(['Date','Alpha']).sort_index()
        # safety clamps
        res['VaR'] = np.minimum(res['VaR'].values, -1e-6)
        res['ES']  = np.minimum(res['ES'].values, res['VaR'].values - 1e-6)
        return res
    
    all_results = []
    for a in ALPHAS:
        print(f"\n=== Training & forecasting for alpha={a} ===")
        cfg = ALPHA_CONFIGS[a]
        res_a = run_for_alpha(a, cfg)
        all_results.append(res_a)

    results = pd.concat(all_results).sort_index()
    metrics = evaluate_results_same_as_garch(results, ALPHAS)
    print("\nLSTM metrics (2023 test window, per-α settings):\n", metrics)   

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
        

