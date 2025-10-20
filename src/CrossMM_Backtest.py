import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from IPython.display import display

# Configuration d'affichage et suppression des warnings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore')

# Types et périodes
TYPES = ["SMA", "EMA", "WMA", "HMA"]
PERIODES = [2, 3, 5, 8, 10, 12, 13, 15, 20, 21, 26, 30, 34, 50, 55, 89, 100, 200]
TYPES = ["SMA", "EMA"]

def calculate_ma(series, period, ma_type):
    if period < 1:
        raise ValueError("Period must be >= 1")
    if ma_type == "SMA":
        return series.rolling(window=period).mean()
    elif ma_type == "EMA":
        return series.ewm(span=period, adjust=False).mean()
    elif ma_type == "WMA":
        if period == 1:
            return series.copy()
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    elif ma_type == "HMA":
        if period == 1:
            return series.copy()
        half_length = max(1, int(period / 2))
        sqrt_length = max(1, int(np.sqrt(period)))
        wma_half = calculate_ma(series, half_length, "WMA")
        wma_full = calculate_ma(series, period, "WMA")
        diff = 2 * wma_half - wma_full
        return calculate_ma(diff, sqrt_length, "WMA")
    else:
        raise ValueError(f"Type de moyenne mobile inconnu : {ma_type}")

def backtest_moving_averages(symbol, start_date='2000-01-01', end_date='2025-01-01',
                             initial_capital=10000.0, max_allocation_pct=0.20,
                             ratio_min=1.5, ratio_max=13.0, min_data_days=200):
    print(f"Analyse de {symbol} du {start_date} au {end_date}...")
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    print(f"Données brutes récupérées : {len(data)} lignes")
    if len(data) == 0:
        print("Aucune donnée trouvée")
        return None, None, None

    data = data.dropna()
    print(f"Données nettoyées : {len(data)} lignes")
    if len(data) < min_data_days:
        print("Trop peu de données utilisables après nettoyage pour un backtest fiable.")
        return None, None, None

    buy_hold_return = float(((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100)
    results = []

    total_tests = 0
    for type1 in TYPES:
        for type2 in TYPES:
            for p1 in PERIODES:
                for p2 in PERIODES:
                    if p1 >= p2:
                        continue
                    ratio = p2 / p1
                    if ratio < ratio_min or ratio > ratio_max:
                        continue
                    if len(data) < p2 + 10:
                        continue

                    total_tests += 1
                    df = data.copy()

                    # Calcul des MM selon le type demandé
                    df['MM_Short'] = calculate_ma(df['Close'], p1, type1)
                    df['MM_Long'] = calculate_ma(df['Close'], p2, type2)

                    # Détection des croisements (Signal à la clôture du jour j, exécution ouverture j+1)
                    df['Signal'] = 0
                    df.loc[df['MM_Short'] > df['MM_Long'], 'Signal'] = 1
                    df['Position'] = df['Signal'].diff()

                    # Gestion du capital et des trades
                    current_capital = initial_capital
                    trades = []
                    open_trades = []

                    # Boucle robuste : accès par .iloc pour garantir des scalaires
                    n = len(df)
                    for j in range(n - 1):
                        # ligne j et j+1
                        row = df.iloc[j]
                        next_row = df.iloc[j + 1]

                        # Récupérer Position en scalaire sécurisé
                        pos_val = row.get('Position', np.nan)
                        try:
                            # forcer en float si possible
                            pos_scalar = float(pos_val)
                        except Exception:
                            pos_scalar = np.nan

                        if np.isnan(pos_scalar):
                            buy_signal = False
                            sell_signal = False
                        else:
                            buy_signal = (pos_scalar == 1.0)
                            sell_signal = (pos_scalar == -1.0)

                        # Prix d'exécution (Open du jour suivant) en scalaire sécurisé
                        exec_val = next_row.get('Open', np.nan)
                        try:
                            execution_price = float(exec_val)
                        except Exception:
                            continue  # pas de prix d'exécution

                        # CLÔTURE (FIFO)
                        if sell_signal and open_trades:
                            buy_data = open_trades.pop(0)
                            buy_date = buy_data['buy_date']
                            qty = int(buy_data['quantity'])
                            buy_price = float(buy_data['buy_price'])
                            order_value = float(buy_data['order_value'])

                            sale_value = qty * execution_price
                            profit = sale_value - order_value
                            current_capital += profit + order_value

                            pct_change = float((execution_price - buy_price) / buy_price * 100) if buy_price != 0 else 0.0
                            trades.append({
                                'buy_date': buy_date,
                                'sell_date': df.index[j + 1],
                                'buy_price': buy_price,
                                'sell_price': execution_price,
                                'pct_change': pct_change,
                                'quantity': qty,
                                'profit_€': profit,
                                'is_closed': True
                            })

                        # OUVERTURE
                        elif buy_signal:
                            max_order_value = current_capital * max_allocation_pct
                            quantity_to_buy = np.floor(max_order_value / execution_price)
                            if quantity_to_buy < 1:
                                if execution_price <= current_capital:
                                    quantity_to_buy = 1
                                else:
                                    continue
                            order_value = quantity_to_buy * execution_price
                            current_capital -= order_value
                            open_trades.append({
                                'buy_date': df.index[j + 1],
                                'buy_price': float(execution_price),
                                'quantity': int(quantity_to_buy),
                                'order_value': float(order_value)
                            })

                    # Clôture des positions ouvertes à la fin du backtest
                    if open_trades:
                        last_date = df.index[-1]
                        last_price = float(df['Close'].iloc[-1])
                        for buy_data in open_trades:
                            buy_date = buy_data['buy_date']
                            qty = int(buy_data['quantity'])
                            buy_price = float(buy_data['buy_price'])
                            order_value = float(buy_data['order_value'])

                            final_sale_value = qty * last_price
                            profit = final_sale_value - order_value
                            current_capital += final_sale_value

                            pct_change = float((last_price - buy_price) / buy_price * 100) if buy_price != 0 else 0.0
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

                    if len(trades) == 0:
                        continue

                    # Statistiques
                    total_return = ((current_capital - initial_capital) / initial_capital) * 100
                    total_profit_eur = current_capital - initial_capital
                    gains = [t['pct_change'] for t in trades if t['pct_change'] > 0]
                    losses = [t['pct_change'] for t in trades if t['pct_change'] < 0]
                    win_rate = (len(gains) / len(trades)) * 100 if len(trades) > 0 else 0.0
                    avg_gain = float(np.mean(gains)) if len(gains) > 0 else 0.0
                    max_gain = float(max(gains)) if len(gains) > 0 else 0.0
                    min_gain = float(min(gains)) if len(gains) > 0 else 0.0
                    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
                    max_loss = float(max(losses)) if len(losses) > 0 else 0.0
                    min_loss = float(min(losses)) if len(losses) > 0 else 0.0

                    results.append({
                        'Type_Court': type1,
                        'Type_Long': type2,
                        'MM_Court': int(p1),
                        'MM_Long': int(p2),
                        'Combinaison': f'{type1}{p1}/{type2}{p2}',
                        'Nb_Trades': int(len(trades)),
                        'Capital_Initial_€': float(initial_capital),
                        'Capital_Final_€': float(current_capital),
                        'Profit_Total_€': float(total_profit_eur),
                        'Taux_Reussite_%': float(win_rate),
                        'Rendement_Total_%': float(total_return),
                        'Gain_Moyen_%': avg_gain,
                        'Gain_Max_%': max_gain,
                        'Gain_Min_%': min_gain,
                        'Perte_Moyenne_%': avg_loss,
                        'Perte_Max_%': max_loss,
                        'Perte_Min_%': min_loss,
                        'trades': trades
                    })

    print(f"Total de combinaisons testées (filtrées) : {total_tests}")
    if not results:
        print("Aucun résultat valide (pas assez de données ou aucun trade généré).")
        return None, None, None

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Rendement_Total_%', ascending=False).reset_index(drop=True)
    best = results_df.iloc[0].to_dict()
    print(f"Analyse terminée: {len(results_df)} combinaisons testées (avec trades)")
    return results_df, best, buy_hold_return

def display_results(results_df, best_strategy, buy_hold_return):
    print("\n" + "="*100)
    print("CLASSEMENT DES STRATEGIES (par Rendement Total basé sur le Capital)")
    print("="*100 + "\n")

    display_df = results_df[['Combinaison', 'Nb_Trades', 'Capital_Final_€', 'Profit_Total_€',
                             'Taux_Reussite_%', 'Rendement_Total_%', 'Gain_Moyen_%', 'Gain_Max_%',
                             'Perte_Moyenne_%', 'Perte_Max_%']].copy()

    for col in ['Taux_Reussite_%', 'Rendement_Total_%', 'Gain_Moyen_%', 'Gain_Max_%', 'Perte_Moyenne_%', 'Perte_Max_%']:
         display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")

    for col in ['Capital_Final_€', 'Profit_Total_€']:
         display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}€".replace(",", "X").replace(".", ",").replace("X", " "))

    display(display_df)

    print("\n" + "="*100)
    print("DETAILS DE LA MEILLEURE STRATEGIE")
    print("="*100)
    print(f"Combinaison: {best_strategy['Combinaison']}")
    print(f"Nombre de trades: {int(best_strategy['Nb_Trades'])}")
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

    for i, trade in enumerate(trades, 1):
        result_type = "GAIN" if trade['profit_€'] > 0 else "PERTE"
        trades_data.append({
            'Trade': f"#{i}",
            'Statut': "Fermé" if trade['is_closed'] else "Ouvert (Fin Test)",
            'Résultat': result_type,
            'Quantité': int(trade['quantity']),
            'Date Achat': trade['buy_date'].strftime("%Y-%m-%d"),
            'Prix Achat': f"{trade['buy_price']:.2f}€",
            'Date Vente': trade['sell_date'].strftime("%Y-%m-%d"),
            'Prix Vente': f"{trade['sell_price']:.2f}€",
            'Profit': f"{trade['profit_€']:,.2f}€".replace(",", "X").replace(".", ",").replace("X", " "),
            'Variation': f"{trade['pct_change']:.2f}%",
        })
    trades_df = pd.DataFrame(trades_data)
    display(trades_df)

# ================================================================
# EXECUTION (paramètres)
# ================================================================
SYMBOL = 'mc.PA'  # exemple: 'AAPL', 'TSLA', '^FCHI', 'MC.PA' ou 'mc.PA'
START_DATE = '2020-01-01'
END_DATE = '2025-10-01'
CAPITAL_INITIAL = 10000.0
ALLOCATION_MAX = 0.20

print(f"--- PARAMETRES DE GESTION DU CAPITAL ---")
print(f"Capital de départ: {CAPITAL_INITIAL:,.2f}€".replace(",", "X").replace(".", ",").replace("X", " "))
print(f"Allocation max par trade: {ALLOCATION_MAX*100:.0f}%")
print("------------------------------------------\n")

results_df, best_strategy, buy_hold_return = backtest_moving_averages(
    symbol=SYMBOL,
    start_date=START_DATE,
    end_date=END_DATE,
    initial_capital=CAPITAL_INITIAL,
    max_allocation_pct=ALLOCATION_MAX,
    ratio_min=1.5,
    ratio_max=13.0,
    min_data_days=200
)

if results_df is not None:
    display_results(results_df, best_strategy, buy_hold_return)