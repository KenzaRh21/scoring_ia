# debug_dar_agro.py
import pymysql
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv(override=True)

DB_USER  = os.getenv("DB_USER",   "root")
DB_PORT  = int(os.getenv("DB_PORT", 4306))
DB_PASS  = os.getenv("DB_PASS",   "")
DB_NAME  = os.getenv("DB_NAME",   "dolidb")
PREFIX   = os.getenv("DB_PREFIX", "llxpx_")

conn   = pymysql.connect(
    host="127.0.0.1", user=DB_USER,
    password=DB_PASS if DB_PASS else None,
    database=DB_NAME, port=DB_PORT, charset='utf8mb4'
)
cursor = conn.cursor(pymysql.cursors.DictCursor)

# 1. Factures Dar Agro
cursor.execute(f"""
    SELECT f.rowid, f.ref, f.paye, f.fk_statut,
           f.date_lim_reglement,
           s.nom
    FROM {PREFIX}facture f
    JOIN {PREFIX}societe s ON f.fk_soc = s.rowid
    WHERE s.nom LIKE '%Dar Agro%'
    AND f.entity = 1
    ORDER BY f.datef
""")
factures = cursor.fetchall()
print("=== FACTURES DAR AGRO ===")
for r in factures:
    print(r)

# 2. Paiements pour ces factures
ids = [str(r['rowid']) for r in factures]
if ids:
    cursor.execute(f"""
        SELECT pf.fk_facture, p.datep, p.amount
        FROM {PREFIX}paiement p
        JOIN {PREFIX}paiement_facture pf ON p.rowid = pf.fk_paiement
        WHERE pf.fk_facture IN ({','.join(ids)})
        ORDER BY pf.fk_facture
    """)
    paiements = cursor.fetchall()
    print("\n=== PAIEMENTS ===")
    for r in paiements:
        print(r)

cursor.close()
conn.close()