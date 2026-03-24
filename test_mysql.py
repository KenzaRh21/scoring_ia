# test_env.py
from dotenv import load_dotenv, find_dotenv
import os

print("Fichier .env trouvé :", find_dotenv())
load_dotenv(override=True)
print("DB_USER:", os.getenv("DB_USER"))
print("DB_PASS:", repr(os.getenv("DB_PASS")))
print("DB_PORT:", os.getenv("DB_PORT"))