Projet CrossMM Backtest

Structure initiale fournie:
- crossmm_backtest.py (script CLI à implémenter)
- indicators/ma.py (moyennes mobiles)
- engine/ (backtester, vector_backtest, money_manager)
- metrics/ (performance)
- utils/ (config, utilitaires)
- tests/ (pytest)

Premier objectif livré:
- Implémentation des moyennes mobiles (SMA, EMA, WMA, HMA) dans indicators/ma.py
- Tests unitaires correspondants (tests/test_ma.py)
- Exemple de config (config.json.example)
- requirements.txt et ce README minimal

Exécution des tests:
1. Créer et activer un environnement virtuel Python 3.9+.
2. Installer dépendances:
   pip install -r requirements.txt
3. Lancer pytest:
   pytest -q

Prochaine étape:
- Implémenter utils/load_config et MoneyManager puis vector_backtest (PR 3 et PR 4).