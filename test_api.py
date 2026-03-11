from dotenv import load_dotenv
import os, requests, pandas as pd

load_dotenv()

URL = os.getenv("DOLIBARR_URL")
KEY = os.getenv("DOLIBARR_KEY")

HEADERS = {
    "DOLAPIKEY": KEY,
    "Accept": "application/json"   # ← force JSON au lieu de XML
}

def get_all_pages(endpoint, params={}):
    results = []
    page = 0
    base = URL.rstrip("/")
    while True:
        p = {"limit": 100, "page": page, **params}
        r = requests.get(f"{base}/api/index.php/{endpoint}", headers=HEADERS, params=p)
        
        # Dolibarr retourne 404 quand il n'y a plus de pages — c'est normal
        if r.status_code == 404:
            break
        
        r.raise_for_status()  # plante uniquement sur les vraies erreurs
        batch = r.json()
        if not batch:
            break
        results.extend(batch)
        print(f"  {endpoint} — page {page} : {len(batch)} elements")
        page += 1
    return results

# --- Test factures ---
print("\n=== FACTURES ===")
factures = get_all_pages("invoices")
df_fact = pd.DataFrame(factures)
print(f"Total factures : {len(df_fact)}")
print("Colonnes disponibles :")
print([c for c in df_fact.columns if df_fact[c].notna().any()])

# --- Test tiers ---
print("\n=== TIERS / CLIENTS ===")
tiers = get_all_pages("thirdparties")
df_soc = pd.DataFrame(tiers)
print(f"Total clients : {len(df_soc)}")
print("Colonnes disponibles :")
print([c for c in df_soc.columns if df_soc[c].notna().any()])

# --- Test paiements sur une facture payee ---
payees = df_fact[df_fact['paye'].astype(str) == '1']
if len(payees) > 0:
    sample_id = payees.iloc[0]['id']
    print(f"\n=== PAIEMENT facture {sample_id} ===")
    r = requests.get(f"{URL}/api/index.php/invoices/{sample_id}/payments", headers=HEADERS)
    print(r.json())
else:
    print("\nAucune facture payee trouvee pour tester les paiements")

# --- Apercu des champs cles ---
print("\n=== CHAMPS CLES FACTURES (5 premieres lignes) ===")
cols_check = ['id','ref','socid','date','date_lim_reglement','total_ttc',
              'statut','paye','mode_reglement_id','cond_reglement_id']
cols_present = [c for c in cols_check if c in df_fact.columns]
print(df_fact[cols_present].head())