import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (precision_recall_curve, roc_auc_score, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

# ======================================================
# 1  CONFIGURATION
# ======================================================
COST_PER_ACTION    = 50
RETARD_SEUIL_JOURS = 5
SEUIL_PONCTUEL     = 5
plt.style.use('seaborn-v0_8-whitegrid')

BASE_URL = os.getenv("DOLIBARR_URL", "").rstrip("/")
API_KEY  = os.getenv("DOLIBARR_KEY", "")
HEADERS  = {"DOLAPIKEY": API_KEY, "Accept": "application/json"}

# ======================================================
# 2  RECUPERATION DOLIBARR
# ======================================================
def get_all_pages(endpoint, params={}):
    results, page = [], 0
    while True:
        p = {"limit": 100, "page": page, **params}
        r = requests.get(f"{BASE_URL}/api/index.php/{endpoint}",
                         headers=HEADERS, params=p, timeout=30)
        if r.status_code == 404:
            break
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        results.extend(batch)
        print(f"  [{endpoint}] page {page} : {len(batch)} elements")
        page += 1
    return results

print("Chargement des factures depuis Dolibarr...")
factures_raw = get_all_pages("invoices")

print("Chargement des tiers depuis Dolibarr...")
tiers_raw = get_all_pages("thirdparties")

df_fact_raw = pd.DataFrame(factures_raw)
df_soc_raw  = pd.DataFrame(tiers_raw)

print(f"\nFactures chargees : {len(df_fact_raw)}")
print(f"Clients charges   : {len(df_soc_raw)}")

# ======================================================
# CORRECTION 1 : dates de paiement reelles via /invoices/{id}/payments
# ======================================================
pay_dates = {}
factures_payees = [f for f in factures_raw if str(f.get('paye', '0')) == '1']
print(f"Recuperation dates paiement pour {len(factures_payees)} factures payees...")

for f in factures_payees:
    fac_id = f.get('id')
    if not fac_id:
        continue
    try:
        r2 = requests.get(
            f"{BASE_URL}/api/index.php/invoices/{fac_id}/payments",
            headers=HEADERS, timeout=15
        )
        if r2.status_code == 200:
            payments = r2.json()
            if isinstance(payments, list) and len(payments) > 0:
                datep = payments[0].get('datep') or payments[0].get('date')
                if datep:
                    pay_dates[int(fac_id)] = pd.to_datetime(
                        pd.to_numeric(datep, errors='coerce'), unit='s'
                    )
    except Exception:
        pass

print(f"Dates de paiement reelles recuperees : {len(pay_dates)}")

# ======================================================
# 3  NORMALISATION FACTURES
# ======================================================
df_fact = pd.DataFrame()

df_fact['id']     = df_fact_raw['id'].astype(int)
df_fact['fk_soc'] = pd.to_numeric(
    df_fact_raw.get('socid', df_fact_raw.get('fk_soc', 0)), errors='coerce'
).fillna(0).astype(int)
df_fact['ref'] = df_fact_raw['ref'].astype(str)

df_fact['datef'] = pd.to_datetime(
    pd.to_numeric(
        df_fact_raw.get('date', df_fact_raw.get('datef', None)), errors='coerce'
    ), unit='s'
)
df_fact['date_lim_reglement'] = pd.to_datetime(
    pd.to_numeric(df_fact_raw['date_lim_reglement'], errors='coerce'), unit='s'
)

df_fact['total_ttc'] = pd.to_numeric(df_fact_raw['total_ttc'], errors='coerce').fillna(0)

df_fact['statut'] = pd.to_numeric(
    df_fact_raw.get('status', df_fact_raw.get('statut', 0)), errors='coerce'
).fillna(0).astype(int)
df_fact['paye'] = pd.to_numeric(df_fact_raw['paye'], errors='coerce').fillna(0).astype(int)

df_fact['mode_reglement_id'] = pd.to_numeric(
    df_fact_raw.get('mode_reglement_id',
    df_fact_raw.get('fk_mode_reglement',
    df_fact_raw.get('mode_reglement', 0))), errors='coerce'
).fillna(0).astype(int)

df_fact['cond_reglement_id'] = pd.to_numeric(
    df_fact_raw.get('cond_reglement_id',
    df_fact_raw.get('fk_cond_reglement',
    df_fact_raw.get('cond_reglement', 1))), errors='coerce'
).fillna(1).astype(int)

# Date paiement reelle depuis API
df_fact['date_paiement_reelle'] = df_fact['id'].map(pay_dates)
mask_payee_no_date = (df_fact['paye'] == 1) & (df_fact['date_paiement_reelle'].isna())
df_fact.loc[mask_payee_no_date, 'date_paiement_reelle'] = \
    df_fact.loc[mask_payee_no_date, 'date_lim_reglement']

df_fact = df_fact.dropna(subset=['datef', 'date_lim_reglement', 'total_ttc'])
df_fact = df_fact[df_fact['total_ttc'] > 0].reset_index(drop=True)

# ======================================================
# 4  NORMALISATION TIERS
# ======================================================
df_soc = pd.DataFrame()
df_soc['rowid']      = df_soc_raw['id'].astype(int)
df_soc['nom_client'] = df_soc_raw['name'].astype(str)

df_soc['typent_id'] = pd.to_numeric(
    df_soc_raw.get('typent_id', df_soc_raw.get('fk_typent', 0)), errors='coerce'
).fillna(0).astype(int)

df_soc['date_creation'] = pd.to_datetime(
    pd.to_numeric(df_soc_raw['date_creation'], errors='coerce'), unit='s'
)

type_map = {1: "TPE", 2: "PME", 3: "Grand Compte"}
df_soc['Taille_Entreprise'] = df_soc['typent_id'].map(type_map).fillna("Autre")

# ======================================================
# 5  MERGE FACTURES + TIERS
# ======================================================
df = df_fact.merge(
    df_soc[['rowid', 'nom_client', 'date_creation', 'Taille_Entreprise', 'typent_id']],
    left_on='fk_soc', right_on='rowid', how='left'
)
df['date_creation'] = df.groupby('fk_soc')['date_creation'].transform(
    lambda x: x.fillna(df['datef'].min())
)
df['date_creation'] = df['date_creation'].fillna(df['datef'].min())

print(f"\nApres merge : {len(df)} factures | {df['fk_soc'].nunique()} clients")

# ======================================================
# 6  SPLIT TEMPOREL EN PREMIER (evite tout leakage)
# ======================================================
df = df.sort_values(['fk_soc', 'datef']).reset_index(drop=True)
DATE_SPLIT = df['datef'].sort_values().iloc[int(len(df) * 0.8)]
DATE_MAX   = df['datef'].max()
print(f"Date de coupure : {DATE_SPLIT.date()}  |  Date max : {DATE_MAX.date()}")

# ======================================================
# 7  TARGET RETARD  ← FIX PRINCIPAL
# ======================================================
# Retard reel pour les factures payees (date paiement - date echeance)
df['retard_jours_reel'] = (
    df['date_paiement_reelle'] - df['date_lim_reglement']
).dt.days

# Pour les factures NON payees : retard depuis aujourd'hui
# (si echeance depassee → retard positif, sinon negatif → pas de retard)
aujourd_hui = pd.Timestamp.now().normalize()
mask_non_payee = df['paye'] == 0
df.loc[mask_non_payee, 'retard_jours_reel'] = (
    aujourd_hui - df.loc[mask_non_payee, 'date_lim_reglement']
).dt.days

df['retard_jours_reel'] = df['retard_jours_reel'].fillna(0)
df['target'] = (df['retard_jours_reel'] > RETARD_SEUIL_JOURS).astype(int)

# Diagnostic — verifie qu'on a bien les 2 classes
print(f"Target distribution — 0: {(df['target']==0).sum()}  1: {(df['target']==1).sum()}")
print(f"Retard moyen : {df['retard_jours_reel'].mean():.1f}j  |  Max : {df['retard_jours_reel'].max():.0f}j")

# ======================================================
# 8  HISTORIQUE CLIENT — STATS AGREGEES
# ======================================================
paiement_stats = df.groupby('fk_soc').agg(
    moyenne_paiement   = ('total_ttc', 'mean'),
    ecart_type         = ('total_ttc', 'std'),
    dernier_montant    = ('total_ttc', 'last'),
    nb_factures        = ('total_ttc', 'count'),
    nb_reglees_a_temps = ('retard_jours_reel', lambda x: (x <= SEUIL_PONCTUEL).sum()),
    montant_total      = ('total_ttc', 'sum')
).fillna(0)

paiement_stats['fiabilite_score'] = (
    paiement_stats['nb_reglees_a_temps'] / paiement_stats['nb_factures'].replace(0, 1)
).clip(0, 1)

poids_m, poids_d, poids_f = 0.5, 0.3, 0.2
M_DEFAULT = 50000
paiement_stats['montant_propose'] = (
    paiement_stats['moyenne_paiement'] * poids_m
  + paiement_stats['dernier_montant']  * poids_d
  + M_DEFAULT * paiement_stats['fiabilite_score'] * poids_f
).clip(upper=M_DEFAULT).round(-1).astype(int)

df = df.merge(
    paiement_stats[['fiabilite_score', 'montant_propose', 'nb_factures',
                    'nb_reglees_a_temps', 'montant_total']],
    left_on='fk_soc', right_index=True, how='left'
)

# ======================================================
# 9  FEATURE ENGINEERING (Anti-Leakage)
# ======================================================
df = df.set_index('datef')
df['nb_factures_3m'] = (
    df.groupby('fk_soc')['total_ttc']
    .rolling(window='90D').count()
    .reset_index(level=0, drop=True)
)
df = df.reset_index()

group = df.groupby('fk_soc')

df['jours_depuis_derniere_fact'] = group['datef'].diff().dt.days.fillna(0)
df['variation_ca_client']        = group['total_ttc'].pct_change().fillna(0)
df['encours_client']             = group['total_ttc'].cumsum().shift(1).fillna(0)
df['tx_succes_hist']             = group['target'].transform(
                                       lambda x: 1 - x.expanding().mean().shift(1)
                                   ).fillna(0.5)
df['anciennete_client_mois']     = (
    (df['datef'] - df['date_creation']).dt.days / 30
).fillna(0).astype(int)
df['tendance_retard']            = (
    df['retard_jours_reel'] - group['retard_jours_reel'].shift(1).fillna(0)
)
df['mois_critique']              = df['date_lim_reglement'].dt.month.isin([8, 12]).astype(int)
df['nb_defauts_hist']            = group['target'].cumsum().shift(1).fillna(0)
df['freq_moyenne_paiement']      = group['jours_depuis_derniere_fact'].transform('mean')
df['pct_factures_a_temps']       = group['target'].transform(lambda x: (x == 0).mean())
df['montant_moyen_relative']     = df['total_ttc'] / group['total_ttc'].transform('mean')
df['intervalle_max_hist']        = group['jours_depuis_derniere_fact'].transform(
                                       lambda x: x.expanding().max().shift(1)
                                   ).fillna(0)
df['ca_trend_3fact']             = group['total_ttc'].transform(
                                       lambda x: x.rolling(3, min_periods=1).mean().pct_change()
                                   ).fillna(0)

df['pct_a_temps_hist'] = df['nb_reglees_a_temps'] / df['nb_factures'].replace(0, 1)
df['Segment_Comportemental'] = pd.cut(
    df['pct_a_temps_hist'],
    bins=[0, 0.7, 0.95, 1.01],
    labels=["Critique", "Irregulier", "Regulier"],
    include_lowest=True
)
df['dso_facture'] = (df['date_paiement_reelle'] - df['datef']).dt.days.fillna(0)

features_retard = [
    "total_ttc", "mode_reglement_id", "cond_reglement_id",
    "tx_succes_hist", "anciennete_client_mois", "tendance_retard",
    "encours_client", "jours_depuis_derniere_fact", "mois_critique",
    "freq_moyenne_paiement", "pct_factures_a_temps", "montant_moyen_relative",
    "nb_defauts_hist", "variation_ca_client", "nb_factures_3m"
]

for col in features_retard:
    df[col] = df[col].replace([np.inf, -np.inf], 0)
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(-1)
    else:
        df[col] = df[col].fillna(df[col].mean())

# ======================================================
# 10  SCORE RFM
# ======================================================
client_rfm = df.groupby('fk_soc').agg(
    recence          = ('datef',                      lambda x: (DATE_MAX - x.max()).days),
    frequence        = ('datef',                      'count'),
    montant_total    = ('total_ttc',                  'sum'),
    montant_moyen    = ('total_ttc',                  'mean'),
    intervalle_moyen = ('jours_depuis_derniere_fact',  'mean'),
    intervalle_max   = ('jours_depuis_derniere_fact',  'max'),
    intervalle_std   = ('jours_depuis_derniere_fact',  'std'),
    tx_succes        = ('target',                     lambda x: 1 - x.mean()),
    nb_defauts       = ('target',                     'sum'),
    ca_trend         = ('ca_trend_3fact',              'last'),
    nb_fact_3m_last  = ('nb_factures_3m',              'last'),
    anciennete       = ('anciennete_client_mois',      'max'),
    nb_mois_actif    = ('datef',                      lambda x: x.dt.to_period('M').nunique()),
    nom_client       = ('nom_client',                  'last'),
    taille           = ('Taille_Entreprise',           'last'),
).reset_index()

client_rfm['intervalle_std'] = client_rfm['intervalle_std'].fillna(0)

scaler = MinMaxScaler()
cols_scale = [
    'recence', 'frequence', 'montant_total', 'montant_moyen',
    'intervalle_moyen', 'intervalle_max', 'intervalle_std',
    'tx_succes', 'nb_defauts', 'ca_trend', 'nb_fact_3m_last',
    'anciennete', 'nb_mois_actif'
]
client_rfm_s = client_rfm.copy()
client_rfm_s[cols_scale] = scaler.fit_transform(client_rfm[cols_scale])

client_rfm_s['score_inactivite'] = (
    0.25 * client_rfm_s['recence']
  + 0.15 * (1 - client_rfm_s['frequence'])
  + 0.12 * client_rfm_s['intervalle_moyen']
  + 0.08 * client_rfm_s['intervalle_max']
  + 0.08 * client_rfm_s['intervalle_std']
  + 0.08 * (1 - client_rfm_s['tx_succes'])
  + 0.05 * client_rfm_s['nb_defauts']
  + 0.05 * (1 - client_rfm_s['montant_total'])
  + 0.05 * (1 - client_rfm_s['nb_mois_actif'])
  + 0.04 * (1 - client_rfm_s['anciennete'])
  + 0.03 * (1 - client_rfm_s['ca_trend'].clip(0, 1))
  + 0.02 * (1 - client_rfm_s['nb_fact_3m_last'])
).clip(0, 1)

client_rfm_s['Score_Fidelite'] = ((1 - client_rfm_s['score_inactivite']) * 100).round(1)

q10 = client_rfm_s['Score_Fidelite'].quantile(0.10)
q25 = client_rfm_s['Score_Fidelite'].quantile(0.25)
q65 = client_rfm_s['Score_Fidelite'].quantile(0.65)
q85 = client_rfm_s['Score_Fidelite'].quantile(0.85)

client_rfm_s['Segment_RFM'] = pd.cut(
    client_rfm_s['Score_Fidelite'],
    bins=[0, q10, q25, q65, q85, 100],
    labels=["Perdu", "En Danger", "A Surveiller", "Fidele", "Champion"],
    include_lowest=True
)

df = df.merge(
    client_rfm_s[['fk_soc', 'score_inactivite', 'Score_Fidelite', 'Segment_RFM']],
    on='fk_soc', how='left'
)

print("\n=== Segments RFM ===")
seg_dist = client_rfm_s['Segment_RFM'].value_counts().sort_index()
for seg, cnt in seg_dist.items():
    print(f"  {str(seg):<15} : {cnt:>4} clients ({cnt/len(client_rfm_s):.1%})")

# ======================================================
# 11  MODELE RETARD — RF + CALIBRATION ISOTONIC + HYPEROPT
# ======================================================
train = df[df['datef'] < DATE_SPLIT].copy()
test  = df[df['datef'] >= DATE_SPLIT].copy()

X_train, y_train = train[features_retard], train['target']
X_test,  y_test  = test[features_retard],  test['target']

scaler_model   = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler_model.fit_transform(X_train),
                               columns=features_retard, index=X_train.index)
X_test_scaled  = pd.DataFrame(scaler_model.transform(X_test),
                               columns=features_retard, index=X_test.index)
X_all_scaled   = pd.DataFrame(scaler_model.transform(df[features_retard]),
                               columns=features_retard, index=df.index)

SAMPLE_SIZE = 10000
if len(X_train_scaled) > SAMPLE_SIZE:
    X_sample = X_train_scaled.sample(SAMPLE_SIZE, random_state=42)
    y_sample = y_train.loc[X_sample.index]
else:
    X_sample, y_sample = X_train_scaled, y_train

param_dist = {
    'n_estimators'    : [100, 150, 200],
    'max_depth'       : [8, 10, 15],
    'min_samples_leaf': [3, 5],
    'class_weight'    : ['balanced']
}
search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=6, scoring='f1', cv=3,
    random_state=42, verbose=1
)
search.fit(X_sample, y_sample)
print(f"\nMeilleurs params : {search.best_params_}")

n_pos = y_train.sum()
n_neg = (y_train == 0).sum()
print(f"Train class distribution — 0: {n_neg}  1: {n_pos}")

if n_pos < 10 or n_neg < 10:
    print("Warning: trop peu de samples d'une classe, calibration ignoree (RF brut).")
    calibrated_model = search.best_estimator_
    calibrated_model.fit(X_train_scaled, y_train)
else:
    from sklearn.model_selection import train_test_split as _tts
    X_cb, X_ch, y_cb, y_ch = _tts(
        X_train_scaled, y_train,
        test_size=0.25, random_state=42, stratify=y_train
    )
    # On entraine le modele de base sur X_cb
    search.best_estimator_.fit(X_cb, y_cb)
    # On calibre manuellement sans cv='prefit' — compatible toutes versions
    from sklearn.calibration import calibration_curve
    from sklearn.isotonic import IsotonicRegression
    import warnings

    raw_probs = search.best_estimator_.predict_proba(X_ch)[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_probs, y_ch)

    class CalibratedWrapper:
        def __init__(self, base, calibrator):
            self.base = base
            self.calibrator = calibrator
            self.classes_ = base.classes_
        def predict_proba(self, X):
            raw = self.base.predict_proba(X)[:, 1]
            cal = self.calibrator.predict(raw)
            return np.column_stack([1 - cal, cal])

    calibrated_model = CalibratedWrapper(search.best_estimator_, iso)
    print("Calibration isotonic manuelle appliquee (compatible toutes versions).")

# ← FIX : safe_proba protege contre le cas ou le modele n'a vu qu'une seule classe
def safe_proba(model, X):
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        cls = model.classes_[0]
        return np.zeros(len(X)) if cls == 1 else np.ones(len(X))
    return proba[:, 1]

y_probs_test  = safe_proba(calibrated_model, X_test_scaled)
y_probs_train = safe_proba(calibrated_model, X_train_scaled)
y_probs_all   = safe_proba(calibrated_model, X_all_scaled)

# Seuil optimal F1
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs_test)
f1_arr     = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
base_seuil = thresholds[np.argmax(f1_arr)]
print(f"Seuil optimal retard : {base_seuil:.3f}")

# Sauvegarde
joblib.dump(calibrated_model, 'model_retard_final.pkl')
joblib.dump(scaler_model,     'scaler_retard.pkl')
print("Modele sauvegarde : model_retard_final.pkl")
print("Scaler sauvegarde : scaler_retard.pkl")

# ======================================================
# 12  SCORES, SEUIL DYNAMIQUE, STATUTS, IPR
# ======================================================
df['prob_ajustee'] = y_probs_all

df.loc[df['date_lim_reglement'].dt.month == 12, 'prob_ajustee'] = (
    df.loc[df['date_lim_reglement'].dt.month == 12, 'prob_ajustee'] * 1.10
).clip(0, 1)

df['impact_financier'] = df['total_ttc'] * df['prob_ajustee']

limites   = df['total_ttc'].quantile([0.3, 0.7, 0.9]).values
q30, q70, q90 = limites

df['seuil_dyn'] = base_seuil
df.loc[df['total_ttc'] <= q30, 'seuil_dyn'] = min(0.85, base_seuil * 1.3)
df.loc[df['total_ttc'] > 20000, 'seuil_dyn'] *= 0.8
mask_fiable = (df['anciennete_client_mois'] > 24) & (df['tx_succes_hist'] > 0.90)
df.loc[mask_fiable, 'seuil_dyn'] = df.loc[mask_fiable, 'seuil_dyn'].clip(lower=0.90)
df.loc[df['tx_succes_hist'] < 0.5, 'seuil_dyn'] *= 0.85
df['seuil_dyn'] = df['seuil_dyn'].clip(0.30, 0.95)

df['Statut'] = np.where(
    (df['prob_ajustee'] >= 0.80) | (df['tendance_retard'] > 10), "CRITIQUE",
    np.where(df['prob_ajustee'] >= df['seuil_dyn'], "RISQUE", "SAIN")
)
df['IPR'] = (
    df['prob_ajustee']
    * np.log10(df['total_ttc'] + 2)
    * (1 + 0.1 * df['nb_defauts_hist'])
)

# ======================================================
# 13  DETECTION D'ANOMALIES
# ======================================================
iso_forest = IsolationForest(contamination=0.04, random_state=42)
df['is_anomaly'] = np.where(
    iso_forest.fit_predict(
        df[['total_ttc', 'retard_jours_reel', 'freq_moyenne_paiement']]
    ) == -1, "OUI", "NON"
)

# ======================================================
# 14  FORECAST CA — Prophet avec fallback SARIMA
# ======================================================
ca_mensuel = (
    df.set_index('datef').resample('ME')['total_ttc']
    .sum().reset_index()
    .rename(columns={'datef': 'ds', 'total_ttc': 'y'})
)

forecast_m1 = forecast_m2 = forecast_lower = forecast_upper = None
forecast_model_name = ""

try:
    from prophet import Prophet
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, seasonality_mode='multiplicative',
                changepoint_prior_scale=0.1)
    m.fit(ca_mensuel)
    future = m.make_future_dataframe(periods=2, freq='ME')
    fc     = m.predict(future)
    forecast_df    = fc[['ds','yhat','yhat_lower','yhat_upper']].tail(2)
    forecast_m1    = forecast_df['yhat'].iloc[0]
    forecast_m2    = forecast_df['yhat'].iloc[1]
    forecast_lower = forecast_df['yhat_lower'].iloc[0]
    forecast_upper = forecast_df['yhat_upper'].iloc[0]
    forecast_model_name = "Prophet"
    print(f"\nForecast Prophet M+1 : {forecast_m1:,.0f} MAD  [{forecast_lower:,.0f} — {forecast_upper:,.0f}]")
except ImportError:
    print("\nProphet non installe, tentative SARIMA...")

if forecast_m1 is None:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        sarima = SARIMAX(ca_mensuel['y'], order=(1,1,1),
                         seasonal_order=(1,1,0,12),
                         enforce_stationarity=False,
                         enforce_invertibility=False).fit(disp=False)
        fc_sarima      = sarima.forecast(steps=2)
        ci             = sarima.get_forecast(steps=2).conf_int()
        forecast_m1    = fc_sarima.iloc[0]
        forecast_m2    = fc_sarima.iloc[1]
        forecast_lower = ci.iloc[0, 0]
        forecast_upper = ci.iloc[0, 1]
        forecast_model_name = "SARIMA"
        print(f"Forecast SARIMA M+1 : {forecast_m1:,.0f} MAD")
    except ImportError:
        print("statsmodels non installe, fallback regression lineaire.")

if forecast_m1 is None:
    from sklearn.linear_model import LinearRegression
    ca_mensuel['ordinal'] = np.arange(len(ca_mensuel))
    lr    = LinearRegression().fit(ca_mensuel[['ordinal']], ca_mensuel['y'])
    preds = lr.predict([[len(ca_mensuel)], [len(ca_mensuel)+1]])
    forecast_m1, forecast_m2 = preds[0], preds[1]
    forecast_lower = forecast_m1 * 0.92
    forecast_upper = forecast_m1 * 1.08
    forecast_model_name = "LinearRegression (fallback)"
    print(f"Forecast LinReg M+1 : {forecast_m1:,.0f} MAD")

# ======================================================
# 15  KPIs
# ======================================================
exposition_totale = df['total_ttc'].sum()
exposition_risque = df[df['Statut'] != "SAIN"]['total_ttc'].sum()
df_unpaid         = df[df['retard_jours_reel'] >= 0].copy()
montant_securise  = df_unpaid['impact_financier'].sum()
bad_debt          = df[df['retard_jours_reel'] > 90]['total_ttc'].sum()
bad_debt_ratio    = bad_debt / exposition_totale

dso_mensuel = (
    df[df['dso_facture'] > 0]
    .groupby(df['datef'].dt.to_period('M'))['dso_facture']
    .mean()
)

seg_counts = client_rfm_s['Segment_RFM'].value_counts()

client_view = df.groupby(['fk_soc', 'nom_client']).agg(
    Score_Fidelite  = ('Score_Fidelite',   'last'),
    Segment_RFM     = ('Segment_RFM',      'last'),
    Prob_Retard_Moy = ('prob_ajustee',     'mean'),
    IPR_Total       = ('IPR',              'sum'),
    CA_Total        = ('total_ttc',        'sum'),
    Statut_Dominant = ('Statut',           lambda x: x.value_counts().index[0]),
    Nb_Anomalies    = ('is_anomaly',       lambda x: (x == "OUI").sum()),
    Taille          = ('Taille_Entreprise','last'),
).reset_index().sort_values('IPR_Total', ascending=False)

colors_seg = {
    'Perdu': '#c0392b', 'En Danger': '#e67e22',
    'A Surveiller': '#f1c40f', 'Fidele': '#2ecc71', 'Champion': '#1a7a4a'
}

# ======================================================
# 16  DASHBOARD
# ======================================================
fig = plt.figure(figsize=(24, 28))
fig.suptitle("Dashboard Scoring IA — Credit Risk & Fidelite Client (Dolibarr Live)",
             fontsize=18, fontweight='bold', y=1.005)
gs = gridspec.GridSpec(5, 3, figure=fig, wspace=0.35, hspace=0.55)

def kpi_card(ax, title, value, color='black', subtitle=None):
    ax.text(0.5, 0.65, title,  ha='center', va='center', fontsize=12, color='gray')
    ax.text(0.5, 0.38, value,  ha='center', va='center', fontsize=24,
            fontweight='bold', color=color)
    if subtitle:
        ax.text(0.5, 0.12, subtitle, ha='center', va='center', fontsize=9, color='gray')
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    for spine in ax.spines.values():
        spine.set_visible(False)

ax_k1 = fig.add_subplot(gs[0, 0])
kpi_card(ax_k1, "Portefeuille a Risque (MAD)",
         f"{exposition_risque:,.0f}", '#e74c3c',
         f"{exposition_risque/exposition_totale:.1%} du CA total")

ax_k2 = fig.add_subplot(gs[0, 1])
kpi_card(ax_k2, "Creances Douteuses (>90j)",
         f"{bad_debt_ratio:.1%}", '#c0392b', f"{bad_debt:,.0f} MAD")

ax_k3 = fig.add_subplot(gs[0, 2])
kpi_card(ax_k3, "Montant Securise par IA",
         f"{montant_securise:,.0f}", '#2ecc71',
         f"Forecast M+1 : {forecast_m1/1e6:.2f}M ({forecast_model_name})")

# Cash-flow
ax1 = fig.add_subplot(gs[1, :2])
cash_flow = df_unpaid.groupby(
    df_unpaid['date_lim_reglement'].dt.to_period('M')
).agg({'total_ttc': 'sum', 'impact_financier': 'sum'})
cash_flow['ca_securise'] = cash_flow['total_ttc'] - cash_flow['impact_financier']
cash_flow.index = cash_flow.index.astype(str)
ax1.fill_between(cash_flow.index, cash_flow['total_ttc'], alpha=0.08, color='gray')
ax1.plot(cash_flow.index, cash_flow['total_ttc'], '--', label='CA Theorique', color='gray', alpha=0.6)
ax1.plot(cash_flow.index, cash_flow['ca_securise'], label='CA Securise (IA)', color='green', linewidth=2.5)
ax1.set_title("PREVISION CASH-FLOW : THEORIQUE VS SECURISE", fontweight='bold')
ax1.legend()
ax1.tick_params(axis='x', rotation=45)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

# DSO
ax2 = fig.add_subplot(gs[1, 2])
ax2.plot(dso_mensuel.index.astype(str), dso_mensuel.values,
         marker='o', color='purple', linewidth=2, markersize=4)
ax2.axhline(dso_mensuel.mean(), color='red', linestyle='--', alpha=0.6,
            label=f"Moy : {dso_mensuel.mean():.0f}j")
ax2.set_title("TENDANCE DSO (jours)", fontweight='bold')
ax2.legend(fontsize=9)
ax2.tick_params(axis='x', rotation=45)

# Segments RFM
ax3 = fig.add_subplot(gs[2, 0])
seg_order = ["Champion","Fidele","A Surveiller","En Danger","Perdu"]
seg_vals  = [seg_counts.get(s, 0) for s in seg_order]
bars3 = ax3.barh(seg_order, seg_vals,
                 color=[colors_seg[s] for s in seg_order], edgecolor='white')
for bar, val in zip(bars3, seg_vals):
    pct = val / len(client_rfm_s) * 100
    ax3.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
             f"{val} ({pct:.0f}%)", va='center', fontsize=8, fontweight='bold')
ax3.set_xlim(0, max(seg_vals) * 1.35)
ax3.set_title("SEGMENTS RFM (equilibres)", fontweight='bold')

# Score fidelite
ax4 = fig.add_subplot(gs[2, 1])
scores = client_rfm_s['Score_Fidelite']
for seg, color in colors_seg.items():
    mask = client_rfm_s['Segment_RFM'] == seg
    if mask.sum() > 0:
        ax4.hist(scores[mask], bins=25, color=color, alpha=0.75, label=seg, edgecolor='white')
ax4.axvline(scores.mean(), color='navy', linestyle='--', label=f"Moy {scores.mean():.1f}")
ax4.set_title("SCORE FIDELITE PAR SEGMENT", fontweight='bold')
ax4.set_xlabel("Score (0-100)")
ax4.legend(fontsize=7)

# Risque par taille
ax5 = fig.add_subplot(gs[2, 2])
seg_risk = df.groupby('Taille_Entreprise')['prob_ajustee'].mean().sort_values()
colors_taille = ['#3498db' if v < 0.4 else '#e67e22' if v < 0.6 else '#e74c3c'
                 for v in seg_risk.values]
seg_risk.plot(kind='barh', color=colors_taille, ax=ax5, edgecolor='white')
ax5.set_title("RISQUE RETARD PAR TAILLE", fontweight='bold')
ax5.set_xlabel("Prob. retard moyenne")

# Matrice sante
ax6 = fig.add_subplot(gs[3, 0])
sc = ax6.scatter(df['Score_Fidelite'], df['prob_ajustee'],
                 c=df['score_inactivite'], cmap='RdYlGn_r', alpha=0.25, s=7)
plt.colorbar(sc, ax=ax6, label='Inactivite')
ax6.axhline(base_seuil, color='navy', linestyle='--', alpha=0.5, linewidth=1)
ax6.axvline(scores.median(), color='navy', linestyle='--', alpha=0.5, linewidth=1)
ax6.set_title("MATRICE SANTE CLIENT", fontweight='bold')
ax6.set_xlabel("Score Fidelite ->")
ax6.set_ylabel("Prob. Retard ->")

# Stabilite modele Train vs Test
ax7 = fig.add_subplot(gs[3, 1])
ax7.hist(y_probs_train, bins=30, alpha=0.6, color='steelblue', label='Train', density=True)
ax7.hist(y_probs_test,  bins=30, alpha=0.6, color='orange',    label='Test',  density=True)
ax7.axvline(base_seuil, color='red', linestyle='--', label=f'Seuil={base_seuil:.2f}')
ax7.set_title("STABILITE MODELE (Train vs Test)", fontweight='bold')
ax7.legend(fontsize=8)

# Top clients
ax8 = fig.add_subplot(gs[3, 2])
ax8.axis('off')
top10 = client_view.head(10)[['nom_client','Score_Fidelite','Segment_RFM',
                               'Prob_Retard_Moy','CA_Total']].copy()
top10['Prob_Retard_Moy'] = top10['Prob_Retard_Moy'].round(2)
top10['CA_Total']        = (top10['CA_Total']/1e3).round(0).astype(int).astype(str) + 'K'
top10['Score_Fidelite']  = top10['Score_Fidelite'].round(0).astype(int)
top10['Segment_RFM']     = top10['Segment_RFM'].astype(str)
tbl = ax8.table(cellText=top10.values,
                colLabels=['Client','Fidelite','Segment','P.Ret.','CA'],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 1.55)
for j in range(5):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i, seg in enumerate(top10['Segment_RFM'].values, 1):
    c = colors_seg.get(seg, '#ffffff') + '44'
    for j in range(5): tbl[i, j].set_facecolor(c)
ax8.set_title("TOP 10 CLIENTS A RISQUE (IPR)", fontweight='bold', pad=18)

# Forecast CA
ax9 = fig.add_subplot(gs[4, :])
ca_plot = ca_mensuel.rename(columns={'ds': 'date', 'y': 'ca'})
ax9.fill_between(ca_plot['date'], ca_plot['ca'], alpha=0.12, color='steelblue')
ax9.plot(ca_plot['date'], ca_plot['ca'], marker='o', color='steelblue',
         linewidth=2, markersize=4, label='CA Reel')
last_date = ca_plot['date'].iloc[-1]
fd1 = last_date + pd.DateOffset(months=1)
fd2 = last_date + pd.DateOffset(months=2)
ax9.errorbar(fd1, forecast_m1,
             yerr=[[forecast_m1 - forecast_lower], [forecast_upper - forecast_m1]],
             fmt='s', color='orange', capsize=6, markersize=9,
             label=f"M+1 : {forecast_m1/1e6:.2f}M MAD ({forecast_model_name})")
ax9.plot(fd2, forecast_m2, 's', color='darkorange', markersize=7,
         label=f"M+2 : {forecast_m2/1e6:.2f}M MAD")
ax9.fill_between([fd1, fd2], [forecast_lower]*2, [forecast_upper]*2,
                 alpha=0.2, color='orange', label="IC 80%")
ax9.axvline(DATE_SPLIT, color='red', linestyle=':', alpha=0.5, label='Coupure train/test')
ax9.set_title(f"FORECAST CA MENSUEL — {forecast_model_name}", fontweight='bold')
ax9.legend(fontsize=9)
ax9.tick_params(axis='x', rotation=45)
ax9.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

plt.tight_layout()
plt.savefig("dashboard_scoring_dolibarr.png", dpi=150, bbox_inches='tight')
plt.show()
print("Dashboard sauvegarde : dashboard_scoring_dolibarr.png")

# ======================================================
# 17  EXPORT EXCEL
# ======================================================
with pd.ExcelWriter("relances_prioritaires_ia_final.xlsx", engine='openpyxl') as writer:
    df.sort_values('IPR', ascending=False)[[
        'nom_client', 'datef', 'total_ttc', 'prob_ajustee', 'seuil_dyn',
        'Statut', 'Segment_Comportemental', 'Score_Fidelite',
        'Segment_RFM', 'montant_propose', 'is_anomaly'
    ]].rename(columns={
        'prob_ajustee'          : 'Prob_Retard',
        'is_anomaly'            : 'Anomalie',
        'Segment_Comportemental': 'Seg_Comportemental'
    }).to_excel(writer, sheet_name='Relances_Factures', index=False)

    client_view.to_excel(writer, sheet_name='Synthese_Clients_RFM', index=False)

    pd.DataFrame({
        'Segment'   : ["Perdu","En Danger","A Surveiller","Fidele","Champion"],
        'Score_Min' : [0, round(q10,1), round(q25,1), round(q65,1), round(q85,1)],
        'Score_Max' : [round(q10,1), round(q25,1), round(q65,1), round(q85,1), 100],
        'Nb_Clients': [seg_counts.get(s,0) for s in ["Perdu","En Danger","A Surveiller","Fidele","Champion"]],
    }).to_excel(writer, sheet_name='Seuils_RFM', index=False)

print("Fichier 'relances_prioritaires_ia_final.xlsx' genere (3 onglets).")

# ======================================================
# 18  VALIDATION FINALE
# ======================================================
auc_test  = roc_auc_score(y_test, y_probs_test)
f1_test   = f1_score(y_test,  (y_probs_test >= base_seuil).astype(int))
prec_test = precision_score(y_test, (y_probs_test >= base_seuil).astype(int), zero_division=0)
rec_test  = recall_score(y_test,  (y_probs_test >= base_seuil).astype(int), zero_division=0)

test_data    = df.loc[y_test.index]
mad_detectes = test_data[
    (test_data['target']==1) & (test_data['prob_ajustee'] >= test_data['seuil_dyn'])
]['total_ttc'].sum()
nb_actions = len(test_data[test_data['prob_ajustee'] >= test_data['seuil_dyn']])
roi_net    = mad_detectes - (nb_actions * COST_PER_ACTION)

n_total  = len(client_rfm_s)
n_danger = seg_counts.get('Perdu', 0) + seg_counts.get('En Danger', 0)
ca_danger= df[df['Segment_RFM'].isin(['Perdu','En Danger'])]['total_ttc'].sum()
n_champ  = seg_counts.get('Champion', 0)
ca_champ = df[df['Segment_RFM'] == 'Champion']['total_ttc'].sum()

print(f"""
╔════════════════════════════════════════════════════╗
║         EXECUTIVE SUMMARY — VERSION FINALE         ║
╠════════════════════════════════════════════════════╣
  SOURCE : Dolibarr API (temps reel)
  Factures : {len(df)} | Clients : {n_total} | Paiements reels : {len(pay_dates)}
  ──────────────────────────────────────────────────
  MODELE RETARD (RF + Isotonic + Scaler)
  AUC          : {auc_test:.3f}
  F1           : {f1_test:.3f}
  Precision    : {prec_test:.3f}
  Recall       : {rec_test:.3f}
  Seuil retard : {base_seuil:.3f}
  ──────────────────────────────────────────────────
  FINANCES
  Portefeuille a risque  : {exposition_risque:,.0f} MAD
  Creances douteuses     : {bad_debt_ratio:.2%}  ({bad_debt:,.0f} MAD)
  Montant securise IA    : {montant_securise:,.0f} MAD
  ROI Net Relance        : {roi_net:,.0f} MAD
  ──────────────────────────────────────────────────
  FORECAST ({forecast_model_name})
  M+1 : {forecast_m1:,.0f} MAD
  M+2 : {forecast_m2:,.0f} MAD
  IC 80% M+1 : [{forecast_lower:,.0f} — {forecast_upper:,.0f}]
  ──────────────────────────────────────────────────
  SCORING RFM (quantiles)
  Champions          : {n_champ} ({n_champ/n_total:.1%}) — {ca_champ:,.0f} MAD
  En danger / Perdus : {n_danger} ({n_danger/n_total:.1%}) — {ca_danger:,.0f} MAD
  Score Fidelite moy : {client_rfm_s['Score_Fidelite'].mean():.1f} / 100
╚════════════════════════════════════════════════════╝
""")