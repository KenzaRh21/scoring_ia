# scoring_api.py
import warnings

from flask import Flask, jsonify
from flask_cors import CORS
import pymysql
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from dotenv import load_dotenv

load_dotenv(override=True)
app  = Flask(__name__)
CORS(app)

MODEL_PATH             = "model_retard_final.pkl"
SEUIL_PAR_DEFAUT       = 0.436
RETARD_SEUIL_JOURS     = 5     # retard > 5j  → au moins RISQUE
SEUIL_PONCTUEL         = 5
RETARD_CRITIQUE_PAYE   = 30    # retard payé   > 30j → CRITIQUE
RETARD_CRITIQUE_IMPAYE = 60    # retard impayé > 60j → CRITIQUE

DB_HOST  = os.getenv("DB_HOST",   "localhost")
DB_USER  = os.getenv("DB_USER",   "root")
DB_PORT  = int(os.getenv("DB_PORT", 4306))
DB_PASS  = os.getenv("DB_PASS",   "")
DB_NAME  = os.getenv("DB_NAME",   "dolidb")
PREFIX   = os.getenv("DB_PREFIX", "llxpx_")

MOIS_FR = {
    1:"Jan", 2:"Fév", 3:"Mar", 4:"Avr", 5:"Mai", 6:"Jun",
    7:"Jul", 8:"Aoû", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Déc"
}

class CalibratedWrapper:
    def __init__(self, base, calibrator):
        self.base       = base
        self.calibrator = calibrator
        self.classes_   = base.classes_
    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])

print("Chargement du modele...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} introuvable")
calibrated_model = joblib.load(MODEL_PATH)
print("Modele charge OK")

def safe_proba(model, X):
    proba = model.predict_proba(X)
    return proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(X))

def get_connection():
    return pymysql.connect(
        host="127.0.0.1",
        user=DB_USER,
        password=DB_PASS if DB_PASS else None,
        database=DB_NAME,
        port=DB_PORT,
        charset='utf8mb4'
    )
def save_scoring_history(kpis):
    try:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO {PREFIX}scoring_history
                  (date_calcul, nb_critiques, nb_risque, nb_sain,
                   exposition_risque, bad_debt, dso_moyen,
                   nb_anomalies, nb_factures, nb_clients)
                VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                kpis['nb_critiques'],
                kpis['nb_risque'],
                kpis['nb_sain'],
                kpis['exposition_risque'],
                kpis['bad_debt'],
                kpis['dso_moyen'],
                kpis['nb_anomalies'],
                kpis['nb_factures'],
                kpis['nb_clients'],
            ))
            conn.commit()
        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        print(f"[scoring_history] Erreur save: {e}")

# SUPPRIME forecast_client() entièrement et remplace par :
def compute_forecasts_clients(df, client_rfm):
    forecasts_clients = {}
    df_tmp = df.copy()
    df_tmp['mois'] = df_tmp['datef'].dt.to_period('M')
    ca_par_mois = df_tmp.groupby(['fk_soc', 'mois'])['total_ttc'].sum()
    prob_map = client_rfm.set_index('fk_soc')['prob_retard_moy'].to_dict()

    for fk, grp in ca_par_mois.groupby(level=0):
        vals = grp.values.astype(float)
        n    = len(vals)
        prob = float(prob_map.get(fk, 0))

        if n < 3:
            moy  = float(np.mean(vals)) if n > 0 else 0
            m1, m2, fiab = moy, moy, "insuffisant"
        else:
            x   = np.arange(n, dtype=float)
            xm  = x.mean()
            ym  = vals.mean()
            denom = np.dot(x - xm, x - xm)
            if denom == 0:
                m1, m2, fiab = float(ym), float(ym), "moyenne"
            else:
                slope     = float(np.dot(x - xm, vals - ym) / denom)
                intercept = ym - slope * xm
                m1   = max(0, slope * n + intercept)
                m2   = max(0, slope * (n + 1) + intercept)
                fiab = "bonne" if n >= 12 else "approximative"

        forecasts_clients[int(fk)] = {
            "forecast_m1":         round(m1, 0),
            "forecast_m2":         round(m2, 0),
            "fiabilite":           fiab,
            "encours_securise_m1": round(m1 * (1 - prob), 0),
            "encours_securise_m2": round(m2 * (1 - prob), 0),
            "prob_retard_moy":     round(prob, 3),
        }
    return forecasts_clients      
def compute_scoring():

    conn   = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    cursor.execute(f"""
        SELECT rowid AS id, ref, fk_soc AS socid,
               datef AS date, date_lim_reglement,
               total_ttc, paye, fk_statut AS statut,
               fk_mode_reglement AS mode_reglement_id,
               fk_cond_reglement AS cond_reglement_id
        FROM {PREFIX}facture
        WHERE entity=1 AND total_ttc>0
          AND datef IS NOT NULL
          AND fk_statut IN (0, 1, 2)
    """)
    factures = cursor.fetchall()

    cursor.execute(f"""
        SELECT rowid AS id, nom AS name, fk_typent,
               datec AS date_creation, code_client, town
        FROM {PREFIX}societe
        WHERE client=1 AND entity=1
    """)
    clients = cursor.fetchall()

    cursor.execute(f"""
        SELECT pf.fk_facture AS id, MIN(p.datep) AS date_paiement
        FROM {PREFIX}paiement p
        JOIN {PREFIX}paiement_facture pf ON p.rowid = pf.fk_paiement
        GROUP BY pf.fk_facture
    """)
    pay_dates = {
        int(r['id']): pd.to_datetime(r['date_paiement'])
        for r in cursor.fetchall()
        if r['date_paiement'] is not None
    }
    cursor.close()
    conn.close()

    # ── aujourd_hui défini ICI — utilisé partout dans la fonction ──
    aujourd_hui = pd.Timestamp.now().normalize()

    df_f = pd.DataFrame(factures)
    df_f['id']     = df_f['id'].astype(int)
    df_f['fk_soc'] = pd.to_numeric(df_f['socid'], errors='coerce').fillna(0).astype(int)
    df_f['datef']  = pd.to_datetime(df_f['date'], errors='coerce')
    df_f['date_lim_reglement'] = pd.to_datetime(df_f['date_lim_reglement'], errors='coerce')
    df_f['total_ttc']         = pd.to_numeric(df_f['total_ttc'], errors='coerce').fillna(0)
    df_f['paye']              = pd.to_numeric(df_f['paye'],      errors='coerce').fillna(0).astype(int)
    df_f['mode_reglement_id'] = pd.to_numeric(df_f['mode_reglement_id'], errors='coerce').fillna(0).astype(int)
    df_f['cond_reglement_id'] = pd.to_numeric(df_f['cond_reglement_id'], errors='coerce').fillna(1).astype(int)

    df_f['date_paiement_reelle'] = df_f['id'].map(pay_dates)

    # ── FIX FALLBACK : facture marquée payée mais sans date de paiement enregistrée
    # On utilise aujourd_hui (et non date_lim_reglement) pour ne pas masquer
    # le retard réel — date_lim_reglement donnerait 0j de retard ce qui est faux
    mask_paye_sans_date = (df_f['paye'] == 1) & (df_f['date_paiement_reelle'].isna())
    df_f.loc[mask_paye_sans_date, 'date_paiement_reelle'] = aujourd_hui

    df_f = df_f.dropna(subset=['datef', 'total_ttc'])
    df_f = df_f[df_f['total_ttc'] > 0].reset_index(drop=True)

    df_s = pd.DataFrame(clients)
    df_s['rowid']         = df_s['id'].astype(int)
    df_s['nom_client']    = df_s['name'].astype(str)
    df_s['typent_id']     = pd.to_numeric(df_s['fk_typent'], errors='coerce').fillna(0).astype(int)
    df_s['date_creation'] = pd.to_datetime(df_s['date_creation'], errors='coerce')
    df_s['Taille']        = df_s['typent_id'].map({1:"TPE", 2:"PME", 3:"Grand Compte"}).fillna("Autre")

    df = df_f.merge(
        df_s[['rowid', 'nom_client', 'date_creation', 'Taille', 'typent_id']],
        left_on='fk_soc', right_on='rowid', how='left'
    )
    df['date_creation'] = df.groupby('fk_soc')['date_creation'].transform(
        lambda x: x.fillna(df['datef'].min())
    ).fillna(df['datef'].min())
    df = df.sort_values(['fk_soc', 'datef']).reset_index(drop=True)

    # ── Calcul retard_jours_reel — forcé float64 ──
    df['retard_jours_reel'] = pd.to_numeric(
        (df['date_paiement_reelle'] - df['date_lim_reglement']).dt.days,
        errors='coerce'
    ).fillna(0.0)
    mask0 = df['paye'] == 0
    df.loc[mask0, 'retard_jours_reel'] = pd.to_numeric(
        (aujourd_hui - df.loc[mask0, 'date_lim_reglement']).dt.days,
        errors='coerce'
    ).fillna(0.0)
    df['retard_jours_reel'] = df['retard_jours_reel'].clip(-30.0, 365.0).astype(float)
    df['target'] = (df['retard_jours_reel'] > RETARD_SEUIL_JOURS).astype(int)

    # ── Stats par client ──
    stats = df.groupby('fk_soc').agg(
        nb_factures        = ('total_ttc', 'count'),
        nb_reglees_a_temps = ('retard_jours_reel', lambda x: (x <= SEUIL_PONCTUEL).sum()),
        montant_total      = ('total_ttc', 'sum'),
        moyenne_paiement   = ('total_ttc', 'mean'),
        dernier_montant    = ('total_ttc', 'last'),
    ).fillna(0)
    stats['fiabilite_score'] = (
        stats['nb_reglees_a_temps'] / stats['nb_factures'].replace(0, 1)
    ).clip(0, 1)

    # ── MODIF 1 : bloquer montant_propose si impayés actifs > 90j ──
    impaye_long_stats = (
        df[(df['paye'] == 0) & (df['retard_jours_reel'] > 90)]
        .groupby('fk_soc')['total_ttc']
        .count()
        .rename('nb_impaye_long')
    )
    stats = stats.join(impaye_long_stats, how='left')
    stats['nb_impaye_long'] = stats['nb_impaye_long'].fillna(0).astype(int)

    stats['montant_propose'] = (
        stats['moyenne_paiement'] * 0.5
      + stats['dernier_montant']  * 0.3
      + 50000 * stats['fiabilite_score'] * 0.2
    ).clip(upper=50000).round(-1).astype(int)

    # Forcer à 0 si le client a des impayés actifs depuis plus de 90 jours
    stats.loc[stats['nb_impaye_long'] > 0, 'montant_propose'] = 0

    df = df.merge(
        stats[['fiabilite_score', 'montant_propose', 'nb_factures',
               'nb_reglees_a_temps', 'montant_total']],
        left_on='fk_soc', right_index=True, how='left'
    )

    # ── Rolling 90j ──
    df = df.set_index('datef')
    df['nb_factures_3m'] = (
        df.groupby('fk_soc')['total_ttc']
        .rolling(window='90D').count()
        .reset_index(level=0, drop=True)
    )
    df = df.reset_index()

    # Réalignement explicite après rolling
    df = df.sort_values(['fk_soc', 'datef']).reset_index(drop=True)

    g = df.groupby('fk_soc')
    df['jours_depuis_derniere_fact'] = g['datef'].diff().dt.days.fillna(0)
    df['variation_ca_client']        = g['total_ttc'].pct_change().fillna(0)
    df['encours_client']             = g['total_ttc'].cumsum().shift(1).fillna(0)
    df['tx_succes_hist']             = g['target'].transform(
                                           lambda x: 1 - x.expanding().mean().shift(1)
                                       ).fillna(0.5)
    df['anciennete_client_mois']     = (
        (df['datef'] - df['date_creation']).dt.days / 30
    ).fillna(0).astype(int)
    df['tendance_retard']            = (
        df['retard_jours_reel'] - g['retard_jours_reel'].shift(1).fillna(0)
    )
    df['mois_critique']              = df['date_lim_reglement'].dt.month.isin([8, 12]).astype(int)
    df['nb_defauts_hist']            = g['target'].cumsum().shift(1).fillna(0)
    df['freq_moyenne_paiement']      = g['jours_depuis_derniere_fact'].transform('mean')
    df['pct_factures_a_temps']       = g['target'].transform(lambda x: (x == 0).mean())
    df['montant_moyen_relative']     = df['total_ttc'] / g['total_ttc'].transform('mean')
    df['ca_trend_3fact']             = g['total_ttc'].transform(
                                           lambda x: x.rolling(3, min_periods=1).mean().pct_change()
                                       ).fillna(0)
    df['pct_a_temps_hist']           = df['nb_reglees_a_temps'] / df['nb_factures'].replace(0, 1)

    df['dso_facture'] = (df['date_paiement_reelle'] - df['datef']).dt.days
    df['dso_facture'] = pd.to_numeric(df['dso_facture'], errors='coerce').fillna(0)

    features = [
        "total_ttc", "mode_reglement_id", "cond_reglement_id",
        "tx_succes_hist", "anciennete_client_mois", "tendance_retard",
        "encours_client", "jours_depuis_derniere_fact", "mois_critique",
        "freq_moyenne_paiement", "pct_factures_a_temps", "montant_moyen_relative",
        "nb_defauts_hist", "variation_ca_client", "nb_factures_3m"
    ]
    for col in features:
        df[col] = df[col].replace([np.inf, -np.inf], 0)
        df[col] = df[col].fillna(df[col].mean() if df[col].dtype != 'object' else -1)

    df['prob_retard'] = safe_proba(calibrated_model, df[features])

    q30 = df['total_ttc'].quantile(0.30)
    df['seuil_dyn'] = SEUIL_PAR_DEFAUT
    df.loc[df['total_ttc'] <= q30, 'seuil_dyn'] = min(0.85, SEUIL_PAR_DEFAUT * 1.3)
    df.loc[df['total_ttc'] > 20000, 'seuil_dyn'] *= 0.8
    mask_fiable = (df['anciennete_client_mois'] > 24) & (df['tx_succes_hist'] > 0.90)
    df.loc[mask_fiable, 'seuil_dyn'] = df.loc[mask_fiable, 'seuil_dyn'].clip(lower=0.90)
    df.loc[df['tx_succes_hist'] < 0.5, 'seuil_dyn'] *= 0.85
    df['seuil_dyn'] = df['seuil_dyn'].clip(0.30, 0.95)
    # ── FLAG nouveau client : < 3 mois ET < 3 factures → pas de pénalité récidiviste ──
    mask_nouveau = (
        (df['anciennete_client_mois'] < 3) &
        (df['nb_factures_3m'] < 3)
    )
    df.loc[mask_nouveau, 'seuil_dyn'] = SEUIL_PAR_DEFAUT
    df['is_nouveau_client'] = mask_nouveau.astype(int)

    # ── Recalcul défensif retard_jours_reel juste avant statut_risque ──
    df['retard_jours_reel'] = pd.to_numeric(
        (df['date_paiement_reelle'] - df['date_lim_reglement']).dt.days,
        errors='coerce'
    ).fillna(0.0)
    mask0_recheck = df['paye'] == 0
    df.loc[mask0_recheck, 'retard_jours_reel'] = pd.to_numeric(
        (aujourd_hui - df.loc[mask0_recheck, 'date_lim_reglement']).dt.days,
        errors='coerce'
    ).fillna(0.0)
    df['retard_jours_reel'] = df['retard_jours_reel'].clip(-30.0, 365.0).astype(float)

    # ── statut_risque : intègre retard réel pour payées ET impayées ──
    cond_critique = (
        (df['prob_retard'] >= 0.80)
        | (df['tendance_retard'] > 10)
        | ((df['paye'] >= 1) & (df['retard_jours_reel'] > float(RETARD_CRITIQUE_PAYE)))
        | ((df['paye'] == 0) & (df['retard_jours_reel'] > float(RETARD_CRITIQUE_IMPAYE)))
    )
    cond_risque = (
        (df['prob_retard'] >= df['seuil_dyn'])
        | ((df['paye'] >= 1) & (df['retard_jours_reel'] > float(RETARD_SEUIL_JOURS)))
        | ((df['paye'] == 0) & (df['retard_jours_reel'] > float(RETARD_SEUIL_JOURS)))
    )
    df['statut_risque'] = np.where(
        cond_critique, "CRITIQUE",
        np.where(cond_risque, "RISQUE", "SAIN")
    )

    # ── DEBUG (décommenter pour diagnostiquer) ──
    # print(df[df['nom_client'].str.contains('Dar Agro', na=False)][
    #     ['ref','paye','retard_jours_reel','prob_retard','statut_risque']
    # ].to_string())

    # ── IPR par facture ──
    df['IPR'] = (
        df['prob_retard']
        * np.log1p(df['total_ttc'])
        * (1 + 0.1 * df['nb_defauts_hist'])
    )

    # ── CA Sécurisé ──
    df['impact_financier'] = df['total_ttc'] * df['prob_retard']
    ca_securise = float(df[df['paye'] == 0]['impact_financier'].sum())

    # ── DSO depuis échéance ──
    df['dso_depuis_echeance'] = (df['date_paiement_reelle'] - df['date_lim_reglement']).dt.days
    df['dso_depuis_echeance'] = pd.to_numeric(
        df['dso_depuis_echeance'], errors='coerce'
    ).fillna(0).clip(0, 365)

    # Pour les factures non payées : DSO = retard courant depuis échéance
    mask_impaye_retard = (
        (df['paye'] == 0)
        & (df['dso_depuis_echeance'] == 0)
        & (df['retard_jours_reel'] > 0)
    )
    df.loc[mask_impaye_retard, 'dso_depuis_echeance'] = df.loc[
        mask_impaye_retard, 'retard_jours_reel'
    ]

    dso_moyen = df[df['dso_depuis_echeance'] > 0]['dso_depuis_echeance'].mean()

    # ── IsoForest : fit uniquement sur factures payées ──
    iso_features = df[['total_ttc', 'retard_jours_reel', 'freq_moyenne_paiement']].copy()
    iso_features = iso_features.replace([np.inf, -np.inf], 0).fillna(0)
    mask_paid = df['paye'] >= 1
    iso = IsolationForest(contamination=0.04, random_state=42)
    iso.fit(iso_features[mask_paid])
    df['is_anomalie'] = (iso.predict(iso_features) == -1).astype(int)
    # Impayés avec long retard = comportement attendu, pas une anomalie
    df.loc[(df['paye'] == 0) & (df['retard_jours_reel'] > 90), 'is_anomalie'] = 0
    nb_anomalies = int(df['is_anomalie'].sum())

    # ── Risque par taille ──
    taille_map = {1: "TPE", 2: "PME", 3: "Grand Compte"}
    risque_taille = (
        df.groupby('typent_id')['prob_retard']
        .mean().mul(100).round(1)
    )
    risque_par_taille = [
        {"t": taille_map[int(k)], "v": float(v)}
        for k, v in risque_taille.items()
        if int(k) in taille_map
    ]

    DATE_MAX   = df['datef'].max()
    client_rfm = df.groupby('fk_soc').agg(
        recence          = ('datef',                      lambda x: (DATE_MAX - x.max()).days),
        frequence        = ('datef',                      'count'),
        montant_total    = ('total_ttc',                  'sum'),
        montant_moyen    = ('total_ttc',                  'mean'),
        intervalle_moyen = ('jours_depuis_derniere_fact', 'mean'),
        intervalle_max   = ('jours_depuis_derniere_fact', 'max'),
        intervalle_std   = ('jours_depuis_derniere_fact', 'std'),
        tx_succes        = ('target',                     lambda x: 1 - x.mean()),
        nb_defauts       = ('target',                     'sum'),
        ca_trend         = ('ca_trend_3fact',             'last'),
        nb_fact_3m_last  = ('nb_factures_3m',             'last'),
        anciennete       = ('anciennete_client_mois',     'max'),
        nb_mois_actif    = ('datef',                      lambda x: x.dt.to_period('M').nunique()),
        nom_client       = ('nom_client',                 'last'),
        taille           = ('Taille',                     'last'),
        prob_retard_moy  = ('prob_retard',                'mean'),
        ipr_total        = ('IPR',                        'mean'),
        ipr_max          = ('IPR',                        'max'),
        ipr_sum          = ('IPR',                        'sum'),
        ca_total         = ('total_ttc',                  'sum'),
        is_anomalie      = ('is_anomalie',                'max'),
        is_nouveau       = ('is_nouveau_client',          'max'),   
    ).reset_index()

    client_rfm['intervalle_std'] = client_rfm['intervalle_std'].fillna(0)

    # ── MODIF 2 : impayés actifs > 90j par client pour pénalité score fidélité ──
    client_rfm['impaye_long'] = (
        df[(df['paye'] == 0) & (df['retard_jours_reel'] > 90)]
        .groupby('fk_soc')['total_ttc']
        .count()
        .reindex(client_rfm['fk_soc'].values)
        .values
    )
    client_rfm['impaye_long'] = client_rfm['impaye_long'].fillna(0).astype(int)
    # ── IPR agrégé client — pondéré par encours total réel ──
    client_rfm['ipr_agrege'] = (
        client_rfm['prob_retard_moy']
        * np.log1p(client_rfm['ca_total'])
        * (1 + 0.1 * client_rfm['nb_defauts'])
    ).round(2)
    # ── MODIF 3 : impaye_long inclus dans MinMaxScaler ──
    scaler     = MinMaxScaler()
    cols_scale = [
        'recence', 'frequence', 'montant_total', 'montant_moyen',
        'intervalle_moyen', 'intervalle_max', 'intervalle_std',
        'tx_succes', 'nb_defauts', 'ca_trend', 'nb_fact_3m_last',
        'anciennete', 'nb_mois_actif', 'impaye_long'  # ← ajout
    ]
    rfm_s = client_rfm.copy()
    rfm_s[cols_scale] = scaler.fit_transform(client_rfm[cols_scale])

    # ── score_inactivite : poids redistribués + pénalité impaye_long ──
    # Note : impaye_long utilisé brut (clip 0-3 / 3) car déjà normalisé via scaler
    # Somme des poids : 0.17+0.14+0.09+0.06+0.06+0.17+0.09+0.05+0.05+0.04+0.02+0.01+0.05 = 1.00
    rfm_s['score_inactivite'] = (
        0.17 * rfm_s['recence']
      + 0.14 * (1 - rfm_s['frequence'])
      + 0.09 * rfm_s['intervalle_moyen']
      + 0.06 * rfm_s['intervalle_max']
      + 0.06 * rfm_s['intervalle_std']
      + 0.17 * (1 - rfm_s['tx_succes'])
      + 0.09 * rfm_s['nb_defauts']
      + 0.05 * (1 - rfm_s['montant_total'])
      + 0.05 * (1 - rfm_s['nb_mois_actif'])
      + 0.04 * (1 - rfm_s['anciennete'])
      + 0.02 * (1 - rfm_s['ca_trend'].clip(0, 1))
      + 0.01 * (1 - rfm_s['nb_fact_3m_last'])
      + 0.05 * (client_rfm['impaye_long'].clip(0, 3) / 3)  # pénalité impayés actifs longs
    ).clip(0, 1)
    rfm_s['score_fidelite'] = ((1 - rfm_s['score_inactivite']) * 100).round(1)

    q10 = rfm_s['score_fidelite'].quantile(0.10)
    q25 = rfm_s['score_fidelite'].quantile(0.25)
    q65 = rfm_s['score_fidelite'].quantile(0.65)
    q85 = rfm_s['score_fidelite'].quantile(0.85)

    rfm_s['segment_rfm'] = pd.cut(
        rfm_s['score_fidelite'],
        bins=[0, q10, q25, q65, q85, 100],
        labels=["Perdu", "En Danger", "A Surveiller", "Fidele", "Champion"],
        include_lowest=True
    ).astype(str)

    # ── Distribution score fidélité par segment (FHIST) ──
    bins_hist   = [0, 20, 35, 45, 55, 65, 75, 85, 100]
    labels_hist = ["0–20","20–35","35–45","45–55","55–65","65–75","75–85","85–100"]
    rfm_s['score_bin'] = pd.cut(
        rfm_s['score_fidelite'],
        bins=bins_hist, labels=labels_hist, include_lowest=True
    )
    fhist_raw = (
        rfm_s.groupby(['score_bin', 'segment_rfm'], observed=True)
        .size().unstack(fill_value=0)
        .reindex(labels_hist).fillna(0)
        .reset_index()
    )
    for col in ['Perdu', 'En Danger', 'A Surveiller', 'Fidele', 'Champion']:
        if col not in fhist_raw.columns:
            fhist_raw[col] = 0
    fhist_list = [
        {
            "r":  str(row['score_bin']),
            "pe": int(row.get('Perdu',        0)),
            "da": int(row.get('En Danger',    0)),
            "su": int(row.get('A Surveiller', 0)),
            "fi": int(row.get('Fidele',       0)),
            "ch": int(row.get('Champion',     0)),
        }
        for _, row in fhist_raw.iterrows()
    ]

    # ── Matrice Santé Client ──
    matrice_sante = rfm_s[[
        'fk_soc', 'nom_client', 'score_fidelite',
        'prob_retard_moy', 'score_inactivite', 'segment_rfm', 'taille'
    ]].copy()
    matrice_sante['score_fidelite']   = matrice_sante['score_fidelite'].round(1)
    matrice_sante['prob_retard_moy']  = matrice_sante['prob_retard_moy'].round(3)
    matrice_sante['score_inactivite'] = matrice_sante['score_inactivite'].round(3)
    matrice_sante_list = matrice_sante.to_dict(orient='records')

    # ── Score fidélité moyen par segment ──
    SEG_ORDER = ["Perdu", "En Danger", "A Surveiller", "Fidele", "Champion"]
    score_par_segment = []
    for seg in SEG_ORDER:
        mask_seg = rfm_s['segment_rfm'] == seg
        if mask_seg.sum() > 0:
            score_par_segment.append({
                "segment":         seg,
                "score_moyen":     round(float(rfm_s.loc[mask_seg, 'score_fidelite'].mean()), 1),
                "score_min":       round(float(rfm_s.loc[mask_seg, 'score_fidelite'].min()),  1),
                "score_max":       round(float(rfm_s.loc[mask_seg, 'score_fidelite'].max()),  1),
                "nb_clients":      int(mask_seg.sum()),
                "prob_retard_moy": round(float(rfm_s.loc[mask_seg, 'prob_retard_moy'].mean()), 3),
            })

    exposition_totale = df['total_ttc'].sum()
    exposition_risque = df[df['statut_risque'] != "SAIN"]['total_ttc'].sum()
    bad_debt = df[(df['paye'] == 0) & (df['retard_jours_reel'] > 90)]['total_ttc'].sum()

    ca_mensuel = (
        df.set_index('datef').resample('ME')['total_ttc']
        .sum().reset_index()
    )
    ca_mensuel['ordinal'] = np.arange(len(ca_mensuel))
    lr          = LinearRegression().fit(ca_mensuel[['ordinal']], ca_mensuel['total_ttc'])
    forecast_m1 = max(0, float(lr.predict([[len(ca_mensuel)]])[0]))
    forecast_m2 = max(0, float(lr.predict([[len(ca_mensuel) + 1]])[0]))

    df_ts = df.copy()
    df_ts['mois_periode'] = df_ts['datef'].dt.to_period('M')
    derniers_12 = sorted(df_ts['mois_periode'].unique())[-12:]

    cashflow_mensuel = []
    dso_mensuel      = []

    for periode in derniers_12:
        label  = MOIS_FR.get(periode.month, str(periode.month))
        mask_p = df_ts['mois_periode'] == periode

        ca_facture     = float(df_ts.loc[mask_p, 'total_ttc'].sum())
        ca_encaisse    = float(df_ts.loc[mask_p & (df_ts['paye'] == 1), 'total_ttc'].sum())
        ca_risque_ia   = float(df_ts.loc[mask_p, 'impact_financier'].sum())
        ca_securise_ia = max(0.0, ca_facture - ca_risque_ia)

        cashflow_mensuel.append({
            "m":    label,
            "theo": round(ca_facture     / 1000, 1),
            "enc":  round(ca_encaisse    / 1000, 1),
            "sec":  round(ca_securise_ia / 1000, 1),
        })

        mask_dso = mask_p & (df_ts['dso_facture'] > 0)
        dso_val  = float(df_ts.loc[mask_dso, 'dso_facture'].mean()) if mask_dso.any() else 0.0
        dso_mensuel.append({"m": label, "v": round(dso_val, 1)})

    nb_critiques_clients = int((client_rfm['prob_retard_moy'] >= 0.80).sum())
    nb_risque_clients    = int(
        ((client_rfm['prob_retard_moy'] >= SEUIL_PAR_DEFAUT) &
         (client_rfm['prob_retard_moy'] <  0.80)).sum()
    )
    nb_sain_clients      = int((client_rfm['prob_retard_moy'] < SEUIL_PAR_DEFAUT).sum())
    nb_critiques_fact    = int((df['statut_risque'] == 'CRITIQUE').sum())
    nb_risque_fact       = int((df['statut_risque'] == 'RISQUE').sum())
    nb_sain_fact         = int((df['statut_risque'] == 'SAIN').sum())

    model_metrics = {
        "auc": 0.805, "f1": 0.740, "accuracy": 0.756,
        "precision": 0.669, "recall": 0.828, "mcc": 0.526,
        "seuil": SEUIL_PAR_DEFAUT,
        "vrai_negatifs": 2499, "faux_positifs": 1049,
        "faux_negatifs": 440,  "vrai_positifs": 2120,
    }

    seg_counts = rfm_s['segment_rfm'].value_counts().to_dict()

    top10 = rfm_s.nlargest(10, 'ipr_agrege')[[
        'fk_soc', 'nom_client', 'score_fidelite', 'segment_rfm',
        'prob_retard_moy', 'ipr_total', 'ipr_agrege', 'ca_total', 'taille', 'is_anomalie', 'is_nouveau'
    ]].copy()
    top10['prob_retard_moy'] = top10['prob_retard_moy'].round(3)
    top10['score_fidelite']  = top10['score_fidelite'].round(1)
    top10['ipr_total']       = top10['ipr_total'].round(2)
    top10['ipr_agrege']      = top10['ipr_agrege'].round(2)
    top10['ca_total']        = top10['ca_total'].round(0)
    top10['is_anomalie']     = top10['is_anomalie'].astype(int)
    top10['is_nouveau']      = top10['is_nouveau'].astype(int)
# ── Forecast CA par client ──
    forecasts_clients = compute_forecasts_clients(df, client_rfm)
    return {
        "kpis": {
            "nb_factures":       len(df),
            "nb_clients":        len(rfm_s),
            "exposition_totale": round(float(exposition_totale), 2),
            "exposition_risque": round(float(exposition_risque), 2),
            "pct_risque":        round(float(exposition_risque / exposition_totale * 100), 1),
            "bad_debt":          round(float(bad_debt), 2),
            "bad_debt_ratio":    round(float(bad_debt / exposition_totale * 100), 2),
            "dso_moyen":         round(float(dso_moyen), 1),
            "forecast_m1":       round(forecast_m1, 2),
            "forecast_m2":       round(forecast_m2, 2),
            "ca_securise":       round(ca_securise, 2),
            "nb_anomalies":      nb_anomalies,
            "nb_critiques":      nb_critiques_clients,
            "nb_risque":         nb_risque_clients,
            "nb_sain":           nb_sain_clients,
            "nb_critiques_fact": nb_critiques_fact,
            "nb_risque_fact":    nb_risque_fact,
            "nb_sain_fact":      nb_sain_fact,
        },
        "segments":            seg_counts,
        "top10_risque":        top10.to_dict(orient='records'),
        "seuils_rfm": {
            "q10": round(float(q10), 1),
            "q25": round(float(q25), 1),
            "q65": round(float(q65), 1),
            "q85": round(float(q85), 1),
        },
        "cashflow_mensuel":    cashflow_mensuel,
        "dso_mensuel":         dso_mensuel,
        "model_metrics":       model_metrics,
        "risque_par_taille":   risque_par_taille,
        "fhist":               fhist_list,
        "matrice_sante":       matrice_sante_list,
        "score_par_segment":   score_par_segment,
        "all_clients": rfm_s[[
            'fk_soc', 'nom_client', 'score_fidelite', 'segment_rfm',
            'prob_retard_moy', 'ipr_total', 'ipr_agrege', 'ca_total', 'taille', 'is_anomalie', 'is_nouveau'
        ]].assign(
            prob_retard_moy = lambda x: x['prob_retard_moy'].round(3),
            score_fidelite  = lambda x: x['score_fidelite'].round(1),
            ipr_total       = lambda x: x['ipr_total'].round(2),
            ipr_agrege      = lambda x: x['ipr_agrege'].round(2),
            ca_total        = lambda x: x['ca_total'].round(0),
            is_anomalie     = lambda x: x['is_anomalie'].astype(int),
            is_nouveau      = lambda x: x['is_nouveau'].astype(int),
        ).to_dict(orient='records'),
        "factures_par_client": {
            str(int(fk)): grp[[
                'rowid', 'ref', 'datef', 'date_lim_reglement',
                'total_ttc', 'prob_retard', 'statut_risque',
                'dso_facture', 'dso_depuis_echeance',
                'retard_jours_reel', 'is_anomalie', 'paye'
            ]].assign(
                datef               = lambda x: x['datef'].dt.strftime('%Y-%m-%d'),
                date_lim_reglement  = lambda x: pd.to_datetime(
                    x['date_lim_reglement'], errors='coerce'
                ).dt.strftime('%Y-%m-%d').fillna('—'),
                total_ttc           = lambda x: x['total_ttc'].round(0),
                prob_retard         = lambda x: x['prob_retard'].round(3),
                dso_facture         = lambda x: pd.to_numeric(
                    x['dso_facture'], errors='coerce'
                ).fillna(0).round(0),
                dso_depuis_echeance = lambda x: pd.to_numeric(
                    x['dso_depuis_echeance'], errors='coerce'
                ).fillna(0).round(0),
                retard_jours_reel   = lambda x: pd.to_numeric(
                    x['retard_jours_reel'], errors='coerce'
                ).fillna(0).round(0),
                is_anomalie         = lambda x: x['is_anomalie'].astype(int),
                paye                = lambda x: x['paye'].astype(int),
            ).to_dict(orient='records')
            for fk, grp in df.groupby('fk_soc')
        },
        "forecasts_clients": forecasts_clients,
    }


@app.route('/api/scoring/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model":  MODEL_PATH,
        "seuil":  SEUIL_PAR_DEFAUT,
        "retard_critique_paye":   RETARD_CRITIQUE_PAYE,
        "retard_critique_impaye": RETARD_CRITIQUE_IMPAYE,
        "retard_seuil_jours":     RETARD_SEUIL_JOURS,
    })

# ── Cache mémoire in-process (5 min) ──
_cache = {"data": None, "ts": 0}
CACHE_TTL = 300

@app.route('/api/scoring/dashboard', methods=['GET'])
def dashboard():
    import time
    now = time.time()
    if _cache["data"] and (now - _cache["ts"]) < CACHE_TTL:
        return jsonify({"status": "ok", "data": _cache["data"]})
    try:
        data = compute_scoring()
        _cache["data"] = data
        _cache["ts"]   = now
        save_scoring_history(data["kpis"])  # ← sauvegarde snapshot
        return jsonify({"status": "ok", "data": data})
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error", "message": str(e), "detail": traceback.format_exc()
        }), 500

@app.route('/api/scoring/relance', methods=['POST'])
def relance():
    try:
        from flask import request
        from datetime import datetime
        data    = request.get_json()
        fk_soc  = data.get('fk_soc')
        nom     = data.get('nom_client')
        montant = data.get('ca_total')
        prob    = data.get('prob_retard_moy')

        # Validation basique fk_soc
        if not isinstance(fk_soc, int):
            return jsonify({"status": "error", "message": "fk_soc invalide"}), 400

        ref = f"CIQ-{datetime.now().strftime('%Y%m%d%H%M%S')}-{fk_soc}"

        conn   = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO {PREFIX}actioncomm
                  (ref, entity, datec, datep, percent,
                   label, note, fk_soc, fk_user_author, code, status)
                VALUES
                  (%s, 1, NOW(), NOW(), 100,
                   %s, %s, %s, 1, 'AC_OTH_AUTO', 1)
            """, (
                ref,
                f"Relance crédit — {nom}",
                (
                    f"Relance automatique CréditIQ\n"
                    f"Prob. retard: {prob}\n"
                    f"Encours: {montant} MAD\n"
                    f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
                ),
                fk_soc
            ))
            conn.commit()
        finally:
            cursor.close()
            conn.close()

        return jsonify({
            "status":  "ok",
            "message": f"Relance enregistrée pour {nom}",
            "ref":     ref,
        })
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error", "message": str(e), "detail": traceback.format_exc()
        }), 500
@app.route('/api/scoring/history', methods=['GET'])
def history():
    try:
        conn   = get_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute(f"""
            SELECT date_calcul, nb_critiques, nb_risque, nb_sain,
                   exposition_risque, bad_debt, dso_moyen,
                   nb_anomalies, nb_factures, nb_clients
            FROM {PREFIX}scoring_history
            ORDER BY date_calcul DESC
            LIMIT 90
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        for r in rows:
            r['date_calcul'] = r['date_calcul'].strftime('%Y-%m-%d %H:%M')
        return jsonify({"status": "ok", "data": rows})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/scoring/activite', methods=['GET'])
def activite():
    try:
        conn   = get_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute(f"""
            SELECT a.id AS id,
                   a.datec AS date_action,
                   a.label,
                   a.note,
                   a.fk_soc,
                   s.nom AS nom_client
            FROM {PREFIX}actioncomm a
            LEFT JOIN {PREFIX}societe s ON s.rowid = a.fk_soc
            WHERE a.code = 'AC_OTH_AUTO'
            ORDER BY a.datec DESC
            LIMIT 50
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        for r in rows:
            if r['date_action']:
                r['date_action'] = r['date_action'].strftime('%Y-%m-%d %H:%M')
        return jsonify({"status": "ok", "data": rows})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print("API Scoring demarree sur http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)