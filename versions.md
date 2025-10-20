# Historique des versions

Ce fichier retrace les principales étapes de développement du projet **Cross Moving Averages Backtest (CrossMM_Backtest)**.
Chaque version correspond à une évolution notable du code, de la structure ou des fonctionnalités.

## 🧩 V1.0 – Initialisation
- Création du dépôt Git
- Ajout du script d’origine `MM_Backtest_ILT.py` 
- Ajout du `README.md` minimal

---

## ⚙️ V1.1 – Première version du script principal
- Création du fichier `CrossMM_Backtest.py`
- Prise en charge de plusieurs types de moyennes : **SMA**, **EMA** (préparation WMA/HMA)
- Gestion du capital et dimensionnement des ordres (Money Management)
- Exécution des signaux à l’ouverture du jour suivant (`J+1`)
- Affichage des résultats consolidés et comparaison avec la stratégie **Buy & Hold**

---

## 🚀 V1.2 – Version de référence (stable)
- Détection des croisements via `delta` / `delta_prev` pour une meilleure précision
- Nettoyage complet du code et renforcement de la robustesse
- Conversion et typage uniformisés pour éviter les erreurs de scalaires
- Consolidation du Money Management (FIFO, ordres minimum, allocation max)

---