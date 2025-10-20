# Cross Moving Averages Backtest (CrossMM_Backtest)

## 🎯 Objectif
Ce projet a pour but de concevoir un outil de **backtesting de croisements de moyennes mobiles**  
(SMA, EMA, etc.) avec une gestion réaliste du capital et des signaux exécutés à l'ouverture du jour suivant.

Le script permet d’évaluer la performance de différentes combinaisons de moyennes mobiles  
et de les comparer à une stratégie de type **Buy & Hold**.

---

## ⚙️ Fonctionnalités principales
- Test automatisé de multiples combinaisons de moyennes mobiles (MM courtes / longues)
- Support des types de moyennes : **SMA**, **EMA**, **WMA**, **HMA**
- Gestion du capital avec allocation maximale par trade
- Exécution des signaux à l’ouverture du jour suivant
- Clôture automatique des positions en FIFO
- Calcul des gains, pertes, taux de réussite et rendement total
- Comparaison avec la performance d’une stratégie **Buy & Hold**
- Affichage clair et structuré des résultats (classement + historique des trades)

---

## 📂 Structure du projet
BacktestMM/
│
├── readme.md ← Ce fichier de présentation
├── versions.md ← Journal des versions
├── roadmap.md ← Idées et axes d’évolution
├── .gitignore
│
└── src/
├── MM_Backtest_ILT.py ← Script d’origine (référence externe)
└── CrossMM_Backtest.py ← Script principal du projet

---

## 🚀 Utilisation

### 1️⃣ Installation des dépendances
```bash
pip install yfinance pandas numpy ipython
```
### 2️⃣ Exécution du backtest
Depuis la racine du projet :
```bash
python src/CrossMM_Backtest.py
```
### 3️⃣ Résultats
Le script affiche :
- un classement des combinaisons de moyennes mobiles testées,
- le détail de la meilleure stratégie,
- la comparaison avec le Buy & Hold,
- et l’historique complet des trades.

## 📘 Historique
Le développement du script suit une progression structurée documentée dans ./versions.md

## 🧭 Axes d’évolution
Les pistes d’amélioration à venir sont listées dans ./roadmap.md

## 📝 Licence
Ce projet est diffusé à des fins pédagogiques et expérimentales.
Aucune garantie de performance n’est donnée sur les résultats obtenus en conditions réelles.