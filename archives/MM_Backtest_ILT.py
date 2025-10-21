import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def backtest_moving_averages(symbol, start_date='2000-01-01', end_date='2025-01-01'):
    """
    Analyse les croisements de moyennes mobiles pour un symbole donné
    Args:
        symbol: code ISIN (ex: 'MC.PA')
        start_date: Date de début du backtest
        end_date: Date de fin du backtest
    
    Returns :
        DataFrame avec les résultats et dictionnaire de la meilleure stratégie
    """
    # Téléchargement des données
    print(f"Analyse de {symbol} du {start_date} au {end_date}...")
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    if len(data) == 0:
        print("Aucune donnée trouvée")
        return None,None, None
    
    print(f"Données téléchargées: {len(data)} jours")

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
    print(f"\nTest de {len( combinations)} combinaisons.")

    for short_ma, long_ma in combinations:
        if len(data) < long_ma + 10:
            continue
        
        # Calcul des moyennes mobiles
        df = data.copy()
        df['MM_Short'] = df['Close'].rolling(window=short_ma).mean()
        df['MM_Long'] = df['Close'].rolling(window=long_ma).mean()

        # Détection des croisements
        df['Signal'] = 0
        df.loc[df['MM_Short'] > df['MM_Long'], 'Signal'] = 1
        #df.loc[df['MM_Short'] < df['MM_Long'], 'Signal'] = -1
        df['Position'] = df['Signal'].diff()

        # Calcul des trades
        buy_signals = df[df['Position'] == 1].copy()
        sell_signals = df[df['Position'] == -1].copy()

        trades = []
        for buy_idx in buy_signals.index:
            future_sells = sell_signals[sell_signals.index > buy_idx]
            if len(future_sells) > 0:
                sell_idx = future_sells.index[0]
                buy_price = float(df.loc[buy_idx, 'Close'])
                sell_price = float(df.loc[sell_idx, 'Close'])
                pct_change = (sell_price - buy_price) / buy_price * 100
                trades.append({
                    'buy_date': buy_idx,
                    'sell_date': sell_idx,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'pct_change': pct_change
                })
        if len(trades) == 0:
            continue
    
        # Calcul des statistiques
        gains = [t['pct_change'] for t in trades if t['pct_change'] > 0]
        losses = [t['pct_change'] for t in trades if t['pct_change'] < 0]
        
        win_rate = (len(gains) / len(trades)) * 100 if len(trades) > 0 else 0
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        max_gain = max(gains) if len(gains) > 0 else 0
        min_gain = min(gains) if len(gains) > 0 else 0
        
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        max_loss = max(losses) if len(losses) > 0 else 0
        min_loss = min(losses) if len(losses) > 0 else 0

        total_return = sum([t['pct_change'] for t in trades])

        results.append({
            'MM_Court': short_ma,
            'MM_Long' : long_ma ,
            'Combinaison' : f'MM{short_ma}/MM{long_ma}',
            'Nb_Trades' : len(trades),
            'Taux_Reussite_%' : win_rate,
            'Rendement_Total_%' : total_return,
            'Gain_Moyen_%' : avg_gain,
            'Gain_Max_%' : max_gain,
            'Gain_Min_%' : min_gain,
            'Perte_Moyenne_%' : avg_loss,
            'Perte_Max_%' : max_loss,
            'Perte_Min_%' : min_loss,
            'trades' : trades
        })
    
    if not results:
        print("Aucun résultat valide")
        return None, None, None
    
    # Conversion en DataFrame et tri
    results_df = pd.DataFrame( results)
    results_df = results_df.sort_values(by='Taux_Reussite_%', ascending=False)

    # Meilleure stratégie
    best = results_df.iloc[0].to_dict()
    print(f"Analyse terminée: {len( results_df)} combinaisons testées")
    return results_df, best, buy_hold_return


def display_results(results_df, best_strategy, buy_hold_return):
    """Affiche les résultats de manière claire"""
    
    print("\n"+"="*100)
    print("CLASSEMENT DES STRATEGIES (par taux de réussite)")
    print("="*100+"\n")
    
    # Préparer le DataFrame pour affichage
    display_df = results_df[['Combinaison', 'Nb_Trades', 'Taux_Reussite_%', 'Rendement_Total_%', 'Gain_Moyen_%', 'Gain_Max_%', 'Gain_Min_%', 'Perte_Moyenne_%', 'Perte_Max_%']].copy()
    
    # Formatage pour affichage
    for col in display_df.columns:
        if '% ' in col:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
    
    from IPython.display import display
    display(display_df)
    
    # Détails de la meilleure stratégie
    print("\n" + "="*100)
    print("DETAILS DE LA MEILLEURE STRATEGIE")
    print("="*100)
    print(f"Combinaison: {best_strategy['Combinaison']}")
    print(f"Nombre de trades: {int(best_strategy['Nb_Trades'])}")
    print(f"Taux de réussite: {best_strategy['Taux_Reussite_%']:.2f}%")
    print(f"Rendement total: {best_strategy['Rendement_Total_%']:.2f}%")
    print(f"\nGains:")
    print(f" Moyen: {best_strategy['Gain_Moyen_%']:.2f}%")
    print(f" Maximum: {best_strategy['Gain_Max_%']:.2f}%")
    print(f" Minimum: {best_strategy['Gain_Min_%']:.2f}%")
    print(f"\nPertes:")
    print(f" Moyenne: {best_strategy['Perte_Moyenne_%']:.2f}%")
    print(f" Maximum: {best_strategy['Perte_Max_%']:.2f}%")
    print(f" Minimum: {best_strategy['Perte_Min_%']:.2f}%")


    # COMPARAISON BUY & HOLD
    print("\n" + "="*100)
    print("COMPARAISON BUY & HOLD")
    print("="*100 + "\n")
    print(f"Stratégie MM {best_strategy[ 'Combinaison' ]}:{best_strategy[ 'Rendement_Total_%' ]:.2f}%")
    print(f"Buy & Hold (acheter et garder) : {buy_hold_return:.2f}%")
    difference = best_strategy[ 'Rendement_Total_%' ] - buy_hold_return
    if difference > 0:
        print(f"\nx Performance supérieure: +{difference:.2f}% vs Buy & Hold")
    else:
        print(f"\nx Performance inférieure: {difference:.2f}% vs Buy & Hold")
    print("\n" + "="*100)
    print('HISTORIQUE DES CROISEMENTS')
    print("="*100)
    trades = best_strategy['trades']
    trades_data = []

    for i, trade in enumerate(trades, 1) :
        result = "GAIN" if trade[ 'pct_change'] > 0 else "PERTE"
        trades_data.append( {
            'Trade' : f"#{i}",
            'Résultat' : result,
            'Date Achat' : trade[ 'buy_date'].strftime("%Y-%m-%d"),
            'Prix Achat' : f"{trade[ 'buy_price']:.2f}",
            'Date Vente' : trade[ 'sell_date'].strftime("%Y-%m-%d"),
            'Prix Vente' : f"{trade[ 'sell_price']:.2f}",
            'Variation' : f"{trade[ 'pct_change']:.2f}%",
        })
    trades_df = pd.DataFrame(trades_data)
    display(trades_df)


#================================================
# EXECUTION
#================================================

#Modifier ici le symbole que vous voulez analyser
SYMBOL = 'mc.PA' # Exemple: LVMH

# Lancer l'analyse
results_df, best_strategy, buy_hold_return = backtest_moving_averages(
	symbol=SYMBOL,
	start_date='2024-01-01',
	end_date='2025-10-01'
)

# Afficher les résultats
if results_df is not None:
	display_results(results_df, best_strategy, buy_hold_return)