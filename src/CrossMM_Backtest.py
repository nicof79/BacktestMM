import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from IPython.display import display

# Configuration d'affichage pour pandas et suppression des warnings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore')

"""
======================================================
SCRIPT DE BACKTESTING : CROISEMENTS DE MOYENNES MOBILES
======================================================

Objectif :
----------
Effectuer un backtest historique pour évaluer la performance de 18 combinaisons
de croisements de moyennes mobiles (MM) sur un actif donné, en simulant des 
exécutions basées sur le prix d'ouverture du jour suivant (signal j, exécution j+1).
Le script intègre une gestion du capital (Money Management) stricte.

Dépendances :
-------------
- yfinance : Téléchargement des données boursières.
- pandas : Manipulation des données (DataFrame).
- numpy : Opérations mathématiques (np.floor, np.mean).
- datetime : Gestion des dates.
- IPython.display : Affichage formaté des résultats (display).

Règles de Money Management (Money Management) :
--------------------------------------
1. Allocation Maximale : Chaque ordre ne peut engager plus de CAPITAL_INITIAL * ALLOCATION_MAX.
2. Taille Minimale : L'achat d'une seule action est la taille minimale d'ordre.
3. Priorité : Le respect de l'ordre minimum (1 action) prime sur l'allocation maximale, 
   si l'allocation ne permet pas l'achat d'une action.
4. Entrée/Sortie : Les trades sont ouverts/fermés à l'**Open** (prix d'ouverture) du jour suivant le signal.

Fonctions Principales :
-----------------------
- backtest_moving_averages() : Télécharge les données, exécute le backtest sur toutes 
  les combinaisons et retourne les résultats.
- display_results() : Affiche le classement des stratégies, les détails de la meilleure
  performance et la comparaison avec la stratégie Buy & Hold.

Configuration Actuelle :
------------------------
- SYMBOL : Actif à tester (défaut : 'mc.PA' / LVMH).
- START_DATE / END_DATE : Période d'analyse.
- CAPITAL_INITIAL : Capital de départ (€).
- ALLOCATION_MAX : Pourcentage maximum du capital alloué par trade.

"""


def backtest_moving_averages(symbol, start_date='2000-01-01', end_date='2025-01-01', 
                             initial_capital=10000.0, max_allocation_pct=0.20):
    """
    Analyse les croisements de moyennes mobiles (MM) avec exécution à l'OUVERTURE du jour suivant,
    en intégrant la gestion du capital et le dimensionnement des ordres (Money Management).
    
    Args:
        symbol: code ISIN (ex: 'MC.PA')
        start_date: Date de début du backtest
        end_date: Date de fin du backtest
        initial_capital: Capital de départ en euros (€)
        max_allocation_pct: Allocation maximale du capital par trade (ex: 0.20 pour 20%)
    
    Returns :
        DataFrame avec les résultats et dictionnaire de la meilleure stratégie
    """
    # Téléchargement des données
    print(f"Analyse de {symbol} du {start_date} au {end_date}...")
    # S'assurer d'avoir les colonnes Open, High, Low, Close
    data = yf.download(symbol, start=start_date, end=end_date, progress=False) 
    
    if len(data) == 0:
        print("Aucune donnée trouvée")
        return None, None, None
    
    # Suppression des lignes avec des valeurs manquantes (pour le calcul des MM et l'Open)
    data = data.dropna() 
    if len(data) < 200:
         print("Trop peu de données utilisables après nettoyage pour un backtest fiable.")
         return None, None, None

    print(f"Données téléchargées et nettoyées: {len(data)} jours")

    # Calcul du Buy & Hold pour comparaison
    buy_hold_return = float(((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100)
    
    # Combinaisons de MM à tester
    combinations = [
        (2, 12), (3, 10),(5, 10),(5, 15),(5, 20),
        (10, 20), (10, 30),(12, 26),(15, 30),(20, 50),
        (20, 100) ,(20, 200),(50, 100),(50, 200),
        (8, 21), (13, 34), (21, 55), (34, 89)
    ]

    results = []
    print(f"\nTest de {len(combinations)} combinaisons.")

    for short_ma, long_ma in combinations:
        if len(data) < long_ma + 10:
            continue
        
        # Calcul des moyennes mobiles
        df = data.copy()
        df['MM_Short'] = df['Close'].rolling(window=short_ma).mean()
        df['MM_Long'] = df['Close'].rolling(window=long_ma).mean()

        # Détection des croisements
        df['Signal'] = 0
        df.loc[df['MM_Short'] > df['MM_Long'], 'Signal'] = 1 # Signal d'achat
        df['Position'] = df['Signal'].diff()

        # --- GESTION DU CAPITAL ET DES TRADES ---
        current_capital = initial_capital 
        trades = [] 
        open_trades = [] # Liste pour suivre les positions ouvertes

        data_index = df.index
        
        # Itération jusqu'à l'avant-dernier jour pour permettre l'exécution le jour 'j+1'
        for j in range(len(df) - 1): 
            
            date = data_index[j]
            next_date = data_index[j+1]
            
            # ✅ Signaux en booléens scalaires
            buy_signal = (df.loc[date, 'Position'] == 1).item()
            sell_signal = (df.loc[date, 'Position'] == -1).item()
            
            # ✅ Prix d'exécution en float scalaire
            execution_price = df.loc[next_date, 'Open'].item()

            # 1. CLÔTURE DE POSITION (Signal de Vente le jour 'j', Exécution à l'OUVERTURE du jour 'j+1')
            if sell_signal and open_trades:
                # Fermer la plus ancienne position ouverte (FIFO)
                buy_data = open_trades.pop(0) 
                
                buy_date = buy_data['buy_date']
                qty = buy_data['quantity']
                # S'assurer que les prix stockés sont bien des floats
                buy_price = float(buy_data['buy_price']) 
                order_value = float(buy_data['order_value'])
                
                # Calcul de la transaction
                sale_value = qty * execution_price
                profit = sale_value - order_value
                
                # Mise à jour du capital
                current_capital += profit + order_value
                
                # ✅ pct_change en float scalaire
                pct_change = float((execution_price - buy_price) / buy_price * 100)
                
                trades.append({
                    'buy_date': buy_date,
                    'sell_date': next_date, 
                    'buy_price': buy_price,
                    'sell_price': execution_price,
                    'pct_change': pct_change,
                    'quantity': qty,
                    'profit_€': profit,
                    'is_closed': True 
                })

            # 2. OUVERTURE DE POSITION (Signal d'Achat le jour 'j', Exécution à l'OUVERTURE du jour 'j+1')
            elif buy_signal:
                
                # Calcul de la valeur maximale de l'ordre (20% du capital disponible)
                max_order_value = current_capital * max_allocation_pct
                
                # Calcul de la quantité, basé sur le prix d'exécution ('Open' du jour j+1)
                quantity_to_buy = np.floor(max_order_value / execution_price)
                
                # Condition d'ordre minimum
                if quantity_to_buy < 1: 
                    if execution_price <= current_capital:
                        quantity_to_buy = 1 
                    else:
                        continue 

                # Valeur réelle de l'ordre
                order_value = quantity_to_buy * execution_price
                
                # Mise à jour du capital (l'argent est immobilisé)
                current_capital -= order_value 
                
                # Enregistrer la position ouverte
                open_trades.append({
                    'buy_date': next_date, 
                    'buy_price': float(execution_price), # Stocker un float
                    'quantity': quantity_to_buy,
                    'order_value': float(order_value) # Stocker un float
                })

        # --- 3. CLÔTURE DES POSITIONS OUVERTES À LA FIN DU BACKTEST ---
        if open_trades:
            last_date = df.index[-1]
            # S'assurer que last_price est un scalaire float, car il vient d'un iloc[-1]
            last_price = float(df['Close'].iloc[-1])
            
            for buy_data in open_trades:
                buy_date = buy_data['buy_date']
                qty = buy_data['quantity']
                # S'assurer que les prix stockés sont bien des floats
                buy_price = float(buy_data['buy_price'])
                order_value = float(buy_data['order_value'])

                # Valeur de revente à la fin du backtest
                final_sale_value = qty * last_price
                profit = final_sale_value - order_value
                
                # Mise à jour du capital final avec la vente des positions restantes
                current_capital += final_sale_value 
                
                # ✅ pct_change en float scalaire
                pct_change = float((last_price - buy_price) / buy_price * 100)
                
                trades.append({
                    'buy_date': buy_date,
                    'sell_date': last_date,
                    'buy_price': buy_price,
                    'sell_price': last_price,
                    'pct_change': pct_change,
                    'quantity': qty,
                    'profit_€': profit,
                    'is_closed': False 
                })

        # FIN DE BOUCLE : Calcul des statistiques sur la base du capital
        if len(trades) == 0:
            continue
    
        # Calcul du rendement total basé sur le capital final
        total_return = ((current_capital - initial_capital) / initial_capital) * 100
        total_profit_eur = current_capital - initial_capital

        gains = [t['pct_change'] for t in trades if t['pct_change'] > 0]
        losses = [t['pct_change'] for t in trades if t['pct_change'] < 0]
        
        win_rate = (len(gains) / len(trades)) * 100 if len(trades) > 0 else 0
        
        # ✅ CORRECTION DU TYPE DE RETOUR : np.mean sur une liste vide renvoie NaN ou un array vide. 
        # On force le résultat en float(0.0) ou le float du résultat de np.mean
        avg_gain = float(np.mean(gains)) if len(gains) > 0 else 0.0
        max_gain = float(max(gains)) if len(gains) > 0 else 0.0
        min_gain = float(min(gains)) if len(gains) > 0 else 0.0
        
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        max_loss = float(max(losses)) if len(losses) > 0 else 0.0
        min_loss = float(min(losses)) if len(losses) > 0 else 0.0

        results.append({
            'MM_Court': short_ma,
            'MM_Long' : long_ma ,
            'Combinaison' : f'MM{short_ma}/MM{long_ma}',
            'Nb_Trades' : len(trades),
            'Capital_Initial_€' : float(initial_capital), # S'assurer d'avoir un float
            'Capital_Final_€' : float(current_capital),   # S'assurer d'avoir un float
            'Profit_Total_€' : float(total_profit_eur),   # S'assurer d'avoir un float
            'Taux_Reussite_%' : float(win_rate),          # S'assurer d'avoir un float
            'Rendement_Total_%' : float(total_return),    # S'assurer d'avoir un float
            'Gain_Moyen_%' : avg_gain,
            'Gain_Max_%' : max_gain,
            'Gain_Min_%' : min_gain,
            'Perte_Moyenne_%' : avg_loss,
            'Perte_Max_%' : max_loss,
            'Perte_Min_%' : min_loss,
            'trades' : trades
        })
    
    # Vérification critique : S'assurer que la liste 'results' n'est pas vide
    if not results:
        print("Aucun résultat valide (pas assez de données ou aucun trade généré).")
        return None, None, None
    
    # Conversion en DataFrame et tri (Ligne 238 après le déplacement et la vérification)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Rendement_Total_%', ascending=False)

    # Meilleure stratégie
    best = results_df.iloc[0].to_dict()
    print(f"Analyse terminée: {len(results_df)} combinaisons testées")
    return results_df, best, buy_hold_return


# ---
def display_results(results_df, best_strategy, buy_hold_return):
    """Affiche les résultats de manière claire"""
    
    print("\n"+"="*100)
    print("CLASSEMENT DES STRATEGIES (par Rendement Total basé sur le Capital)")
    print("="*100+"\n")
    
    # Préparer le DataFrame pour affichage
    display_df = results_df[['Combinaison', 'Nb_Trades', 'Capital_Final_€', 'Profit_Total_€', 'Taux_Reussite_%', 'Rendement_Total_%', 'Gain_Moyen_%', 'Gain_Max_%', 'Perte_Moyenne_%', 'Perte_Max_%']].copy()
    
    # Formatage pour affichage
    for col in ['Taux_Reussite_%', 'Rendement_Total_%', 'Gain_Moyen_%', 'Gain_Max_%', 'Perte_Moyenne_%', 'Perte_Max_%']:
         display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
         
    for col in ['Capital_Final_€', 'Profit_Total_€']:
         # Correction du formatage pour la lisibilité française (espace pour milliers, virgule pour décimaux)
         display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}€".replace(",", "X").replace(".", ",").replace("X", " ")) 

    display(display_df)
    
    # Détails de la meilleure stratégie
    print("\n" + "="*100)
    print("DETAILS DE LA MEILLEURE STRATEGIE")
    print("="*100)
    print(f"Combinaison: {best_strategy['Combinaison']}")
    print(f"Nombre de trades: {int(best_strategy['Nb_Trades'])}")
    # Utilisation du prénom de l'utilisateur pour le formatage (c'est une adresse sur son PC)
    print(f"Capital Initial: {best_strategy['Capital_Initial_€']:,.2f}€".replace(",", "X").replace(".", ",").replace("X", " "))
    print(f"Capital Final: {best_strategy['Capital_Final_€']:,.2f}€".replace(",", "X").replace(".", ",").replace("X", " "))
    print(f"Profit Total: {best_strategy['Profit_Total_€']:,.2f}€".replace(",", "X").replace(".", ",").replace("X", " "))
    print(f"Rendement total: {best_strategy['Rendement_Total_%']:.2f}%")
    print(f"Taux de réussite: {best_strategy['Taux_Reussite_%']:.2f}%")
    print(f"\nGains (pourcentages par trade):")
    print(f" Moyen: {best_strategy['Gain_Moyen_%']:.2f}%")
    print(f" Maximum: {best_strategy['Gain_Max_%']:.2f}%")
    print(f"\nPertes (pourcentages par trade):")
    print(f" Moyenne: {best_strategy['Perte_Moyenne_%']:.2f}%")
    print(f" Maximum: {best_strategy['Perte_Max_%']:.2f}%")


    # COMPARAISON BUY & HOLD
    print("\n" + "="*100)
    print("COMPARAISON BUY & HOLD")
    print("="*100 + "\n")
    print(f"Stratégie MM {best_strategy['Combinaison']} (Rendement sur Capital) : {best_strategy['Rendement_Total_%']:.2f}%")
    print(f"Buy & Hold (acheter et garder) : {buy_hold_return:.2f}%")
    difference = best_strategy['Rendement_Total_%'] - buy_hold_return
    if difference > 0:
        print(f"\n✅ Performance supérieure: +{difference:.2f}% vs Buy & Hold")
    else:
        print(f"\n❌ Performance inférieure: {difference:.2f}% vs Buy & Hold")
    print("\n" + "="*100)
    print('HISTORIQUE DES TRADES (Meilleure Stratégie)')
    print("="*100)
    
    trades = best_strategy['trades']
    trades_data = []

    for i, trade in enumerate(trades, 1) :
        result_type = "GAIN" if trade['profit_€'] > 0 else "PERTE"
        
        trades_data.append( {
            'Trade' : f"#{i}",
            'Statut' : "Fermé" if trade['is_closed'] else "Ouvert (Fin Test)",
            'Résultat' : result_type,
            'Quantité' : int(trade['quantity']),
            'Date Achat' : trade['buy_date'].strftime("%Y-%m-%d"),
            'Prix Achat' : f"{trade['buy_price']:.2f}€",
            'Date Vente' : trade['sell_date'].strftime("%Y-%m-%d"),
            'Prix Vente' : f"{trade['sell_price']:.2f}€",
            # Correction du formatage ici aussi
            'Profit' : f"{trade['profit_€']:,.2f}€".replace(",", "X").replace(".", ",").replace("X", " "), 
            'Variation' : f"{trade['pct_change']:.2f}%",
        })
    trades_df = pd.DataFrame(trades_data)
    display(trades_df)


#================================================
# EXECUTION
#================================================

# --- CONFIGURATION DU BACKTEST ---
SYMBOL = 'mc.PA' # Symbole (LVMH) ou autre (ex: 'TSLA', '^FCHI' pour le CAC40)
START_DATE = '2024-01-01'
END_DATE = '2025-10-01' # Date future pour un test complet sur 2024 et plus

# --- GESTION DU CAPITAL ---
CAPITAL_INITIAL = 10000.0   # Capital de départ en Euros
ALLOCATION_MAX = 0.20       # 20% du capital max par ordre

# Lancer l'analyse
print(f"--- PARAMETRES DE GESTION DU CAPITAL ---")
print(f"Capital de départ: {CAPITAL_INITIAL:,.2f}€".replace(",", "X").replace(".", ",").replace("X", " "))
print(f"Allocation max par trade: {ALLOCATION_MAX*100:.0f}%")
print("------------------------------------------\n")

results_df, best_strategy, buy_hold_return = backtest_moving_averages(
    symbol=SYMBOL,
    start_date=START_DATE,
    end_date=END_DATE,
    initial_capital=CAPITAL_INITIAL,        
    max_allocation_pct=ALLOCATION_MAX       
)

# Afficher les résultats
if results_df is not None:
    display_results(results_df, best_strategy, buy_hold_return)