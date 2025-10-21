import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from IPython.display import display

# Configuration affichage
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore')

# Paramètres globaux
PERIODES = [2, 3, 5, 8, 10, 12, 13, 15, 20, 21, 26, 30, 34, 50, 55, 89, 100, 144, 200]
MA_TYPES = ["SMA", "EMA", "WMA"]
RATIO_MIN = 1.5
RATIO_MAX = 13.0

def wma(series, period):
    if period < 1:
        raise ValueError("Period must be >= 1")
    weights = np.arange(1, period + 1)
    def calc(x):
        return np.dot(x, weights) / weights.sum()
    return series.rolling(window=period).apply(calc, raw=True)

def calculate_ma(series, period, ma_type):
    if ma_type == "SMA":
        return series.rolling(window=period).mean()
    elif ma_type == "EMA":
        return series.ewm(span=period, adjust=False).mean()
    elif ma_type == "WMA":
        return wma(series, period)
    else:
        raise ValueError(f"Unknown MA type: {ma_type}")

def precompute_mas(df, periods, types):
    """
    Ajoute au DataFrame des colonnes pour chaque combinaison <TYPE>_<PERIOD>.
    Ex : SMA_20, EMA_50, WMA_10
    """
    for t in types:
        for p in periods:
            col = f"{t}_{p}"
            if col in df.columns:
                continue
            df[col] = calculate_ma(df['Close'], p, t)
    return df

def first_valid_index_for_pair(df, col_short, col_long):
    """
    Retourne l'index (position entière dans df) de la première ligne
    où les deux colonnes de MM ne sont pas NaN.
    """
    valid_mask = (~df[col_short].isna()) & (~df[col_long].isna())
    if not valid_mask.any():
        return None
    return int(np.where(valid_mask.values)[0][0])

def backtest_moving_averages(symbol,
                             start_date='2000-01-01',
                             end_date='2025-01-01',
                             initial_capital=10000.0,
                             max_allocation_pct=0.20,
                             ratio_min=RATIO_MIN,
                             ratio_max=RATIO_MAX,
                             min_data_days=200,
                             required_post_days=5):
    """
    Backtest simple de croisements de moyennes mobiles.
    Principales règles :
    - Signal déterminé au Close du jour j (cross entre MM_short et MM_long)
    - Exécution tentée à l'Open du jour j+1 si j+1 existe
    - Si position(s) ouvertes à la fin du dataset, elles sont clôturées au dernier Close (clôture forcée)
    - On enregistre is_closed True pour ventes normales, False pour clôtures forcées
    - On commence à backtester à partir de la première date où les deux MM sont valorisées
    """
    print(f"Analyse de {symbol} du {start_date} au {end_date}...")
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    print(f"Données brutes récupérées : {len(data)} lignes")
    if data is None or len(data) == 0:
        print("Aucune donnée trouvée")
        return None, None, None

    data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    print(f"Données nettoyées : {len(data)} lignes")
    if len(data) < min_data_days:
        print("Trop peu de données utilisables après nettoyage pour un backtest fiable.")
        return None, None, None

    # Pré-calcul de toutes les MM demandées pour éviter recalculs coûteux
    df_full = data.copy()
    df_full = precompute_mas(df_full, PERIODES, MA_TYPES)

    buy_hold_return = float(((df_full['Close'].iloc[-1] - df_full['Close'].iloc[0]) / df_full['Close'].iloc[0]) * 100)
    results = []
    total_tests = 0

    for type_short in MA_TYPES:
        for type_long in MA_TYPES:
            for p1 in PERIODES:
                for p2 in PERIODES:
                    if p1 >= p2:
                        continue
                    ratio = p2 / p1
                    if ratio < ratio_min or ratio > ratio_max:
                        continue

                    short_col = f"{type_short}_{p1}"
                    long_col = f"{type_long}_{p2}"

                    # Vérifier que les colonnes existent
                    if short_col not in df_full.columns or long_col not in df_full.columns:
                        continue

                    # trouver la première date où les deux MM sont valorisées
                    first_valid_pos = first_valid_index_for_pair(df_full, short_col, long_col)
                    if first_valid_pos is None:
                        # aucune date où les deux MA sont valides
                        continue

                    # Exiger au moins required_post_days jours après la première date valide
                    if (len(df_full) - first_valid_pos) < required_post_days:
                        continue

                    total_tests += 1

                    # Travailler sur une vue réduite à partir de la première date valide
                    df = df_full.iloc[first_valid_pos:].copy()
                    df = df[[ 'Open', 'High', 'Low', 'Close', short_col, long_col ]].reset_index()
                    # index original en df['Date']
                    df.rename(columns={'index': 'Date'}, inplace=True)

                    # delta et signal
                    df['delta'] = df[short_col] - df[long_col]
                    df['delta_prev'] = df['delta'].shift(1)

                    # signaux simples (cross at close)
                    bull_cross = (df['delta_prev'] <= 0) & (df['delta'] > 0)
                    bear_cross = (df['delta_prev'] >= 0) & (df['delta'] < 0)

                    df['Signal'] = 0
                    df.loc[bull_cross, 'Signal'] = 1
                    df.loc[bear_cross, 'Signal'] = -1

                    # Simulate trades
                    current_cash = float(initial_capital)
                    trades = []
                    open_trades = []  # liste FIFO d'ordres ouverts

                    n = len(df)
                    for j in range(n):
                        row = df.iloc[j]
                        sig = row.get('Signal', 0)
                        try:
                            sig_scalar = float(sig)
                        except Exception:
                            sig_scalar = 0.0

                        buy_signal = (sig_scalar == 1.0)
                        sell_signal = (sig_scalar == -1.0)

                        # execution on next trading day if exists
                        if j + 1 >= n:
                            # pas de jour suivant pour exécution, on skip l'exécution (sera clôturé forcé en fin)
                            continue

                        next_row = df.iloc[j + 1]
                        exec_open = next_row.get('Open', np.nan)
                        try:
                            execution_price = float(exec_open)
                        except Exception:
                            # impossibilité d'exécuter
                            continue

                        # Vente : fermer plus ancienne position FIFO si présente
                        if sell_signal and open_trades:
                            buy_data = open_trades.pop(0)
                            buy_date = buy_data['buy_date']
                            qty = int(buy_data['quantity'])
                            buy_price = float(buy_data['buy_price'])
                            order_value = float(buy_data['order_value'])

                            sale_value = qty * execution_price
                            profit = sale_value - order_value
                            # remettre order_value + profit -> équivalent au cash après vente
                            current_cash += sale_value

                            pct_change = float((execution_price - buy_price) / buy_price * 100) if buy_price != 0 else 0.0
                            trades.append({
                                'buy_date': buy_date,
                                'sell_date': next_row['Date'],
                                'buy_price': buy_price,
                                'sell_price': execution_price,
                                'pct_change': pct_change,
                                'quantity': qty,
                                'profit_€': profit,
                                'is_closed': True,
                                'forced_close': False
                            })

                        # Achat : ouvrir position en utilisant allocation max par ordre
                        elif buy_signal:
                            max_order_value = current_cash * max_allocation_pct
                            quantity_to_buy = np.floor(max_order_value / execution_price)
                            if quantity_to_buy < 1:
                                # si on ne peut pas acheter au moins une action, tenter 1 si le cash le permet
                                if execution_price <= current_cash:
                                    quantity_to_buy = 1
                                else:
                                    continue
                            order_value = quantity_to_buy * execution_price
                            current_cash -= order_value
                            open_trades.append({
                                'buy_date': next_row['Date'],
                                'buy_price': float(execution_price),
                                'quantity': int(quantity_to_buy),
                                'order_value': float(order_value)
                            })

                    # clôture forcée des positions ouvertes au dernier Close
                    if open_trades:
                        last_date = df['Date'].iloc[-1]
                        last_price = float(df['Close'].iloc[-1])
                        for buy_data in open_trades:
                            buy_date = buy_data['buy_date']
                            qty = int(buy_data['quantity'])
                            buy_price = float(buy_data['buy_price'])
                            order_value = float(buy_data['order_value'])

                            final_sale_value = qty * last_price
                            profit = final_sale_value - order_value
                            current_cash += final_sale_value

                            pct_change = float((last_price - buy_price) / buy_price * 100) if buy_price != 0 else 0.0
                            trades.append({
                                'buy_date': buy_date,
                                'sell_date': last_date,
                                'buy_price': buy_price,
                                'sell_price': last_price,
                                'pct_change': pct_change,
                                'quantity': qty,
                                'profit_€': profit,
                                'is_closed': False,
                                'forced_close': True
                            })

                    if len(trades) == 0:
                        continue

                    # calcul des métriques : réalistes vs totales
                    total_return = ((current_cash - initial_capital) / initial_capital) * 100
                    total_profit_eur = current_cash - initial_capital

                    # séparer trades fermés réellement des clôtures forcées
                    closed_trades = [t for t in trades if t['is_closed']]
                    forced_trades = [t for t in trades if not t['is_closed']]

                    gains = [t['pct_change'] for t in trades if t['pct_change'] > 0]
                    losses = [t['pct_change'] for t in trades if t['pct_change'] < 0]
                    win_rate = (len([t for t in trades if t['pct_change'] > 0]) / len(trades)) * 100 if len(trades) > 0 else 0.0

                    avg_gain = float(np.mean([t for t in gains])) if len(gains) > 0 else 0.0
                    max_gain = float(max(gains)) if len(gains) > 0 else 0.0
                    min_gain = float(min(gains)) if len(gains) > 0 else 0.0
                    avg_loss = float(np.mean([t for t in losses])) if len(losses) > 0 else 0.0
                    max_loss = float(max(losses)) if len(losses) > 0 else 0.0
                    min_loss = float(min(losses)) if len(losses) > 0 else 0.0

                    results.append({
                        'Type_Court': type_short,
                        'Type_Long': type_long,
                        'MM_Court': int(p1),
                        'MM_Long': int(p2),
                        'Combinaison': f'{type_short}{p1}/{type_long}{p2}',
                        'Nb_Trades': int(len(trades)),
                        'Nb_Closed_Trades': int(len(closed_trades)),
                        'Nb_Forced_Closes': int(len(forced_trades)),
                        'Capital_Initial_€': float(initial_capital),
                        'Capital_Final_€': float(current_cash),
                        'Profit_Total_€': float(total_profit_eur),
                        'Taux_Reussite_%': float(win_rate),
                        'Rendement_Total_%': float(total_return),
                        'Gain_Moyen_%': avg_gain,
                        'Gain_Max_%': max_gain,
                        'Gain_Min_%': min_gain,
                        'Perte_Moyenne_%': avg_loss,
                        'Perte_Max_%': max_loss,
                        'Perte_Min_%': min_loss,
                        'trades': trades,
                        'start_backtest_date': df['Date'].iloc[0],
                        'first_valid_index': first_valid_pos
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
    print("\n" + "="*80)
    print("CLASSEMENT DES STRATEGIES (par Rendement Total basé sur le Capital)")
    print("="*80 + "\n")

    display_df = results_df[['Combinaison', 'Nb_Trades', 'Nb_Closed_Trades', 'Nb_Forced_Closes',
                             'Capital_Final_€', 'Profit_Total_€', 'Taux_Reussite_%', 'Rendement_Total_%']].copy()

    for col in ['Taux_Reussite_%', 'Rendement_Total_%']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
    for col in ['Capital_Final_€', 'Profit_Total_€']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}€".replace(",", "X").replace(".", ",").replace("X", " "))

    display(display_df)

    print("\n" + "="*80)
    print("DETAILS DE LA MEILLEURE STRATEGIE")
    print("="*80)
    print(f"Combinaison: {best_strategy['Combinaison']}")
    print(f"Nombre de trades total: {int(best_strategy['Nb_Trades'])}")
    print(f"Nombre de trades fermés naturellement: {int(best_strategy['Nb_Closed_Trades'])}")
    print(f"Nombre de clôtures forcées fin test: {int(best_strategy['Nb_Forced_Closes'])}")
    print(f"Date de début effective du backtest pour cette combinaisons: {best_strategy['start_backtest_date']}")
    print(f"Capital Initial: {best_strategy['Capital_Initial_€']:,.2f}€".replace(",", "X").replace(".", ",").replace("X", " "))
    print(f"Capital Final: {best_strategy['Capital_Final_€']:,.2f}€".replace(",", "X").replace(".", ",").replace("X", " "))
    print(f"Profit Total: {best_strategy['Profit_Total_€']:,.2f}€".replace(",", "X").replace(".", ",").replace("X", " "))
    print(f"Rendement total: {best_strategy['Rendement_Total_%']:.2f}%")
    print(f"Taux de réussite: {best_strategy['Taux_Reussite_%']:.2f}%")
    print("\nComparaison buy & hold vs stratégie:")
    print(f"Stratégie MM {best_strategy['Combinaison']} : {best_strategy['Rendement_Total_%']:.2f}%")
    print(f"Buy & Hold : {buy_hold_return:.2f}%")

    print("\n" + "="*80)
    print("HISTORIQUE DES TRADES (Meilleure Stratégie)")
    print("="*80)
    trades = best_strategy['trades']
    trades_data = []
    for i, trade in enumerate(trades, 1):
        result_type = "GAIN" if trade['profit_€'] > 0 else "PERTE"
        forced = trade.get('forced_close', False)
        trades_data.append({
            'Trade': f"#{i}",
            'Statut': "Fermé" if trade['is_closed'] else "Clôture forcée fin test",
            'Forcée': "Oui" if forced else "Non",
            'Résultat': result_type,
            'Quantité': int(trade['quantity']),
            'Date Achat': trade['buy_date'] if isinstance(trade['buy_date'], str) else trade['buy_date'].strftime("%Y-%m-%d"),
            'Prix Achat': f"{trade['buy_price']:.2f}€",
            'Date Vente': trade['sell_date'] if isinstance(trade['sell_date'], str) else trade['sell_date'].strftime("%Y-%m-%d"),
            'Prix Vente': f"{trade['sell_price']:.2f}€",
            'Profit': f"{trade['profit_€']:,.2f}€".replace(",", "X").replace(".", ",").replace("X", " "),
            'Variation': f"{trade['pct_change']:.2f}%"
        })
    trades_df = pd.DataFrame(trades_data)
    display(trades_df)

# =========================
# Paramètres d'exécution
# =========================
SYMBOL = 'MC.PA'   # exemple : 'AAPL', 'TSLA', '^FCHI', 'MC.PA'
START_DATE = '2020-01-01'
END_DATE = '2025-10-01'
CAPITAL_INITIAL = 10000.0
ALLOCATION_MAX = 0.20

print(f"Capital de départ: {CAPITAL_INITIAL:,.2f}€".replace(",", "X").replace(".", ",").replace("X", " "))
print(f"Allocation max par trade: {ALLOCATION_MAX*100:.0f}%\n")

results_df, best_strategy, buy_hold_return = backtest_moving_averages(
    symbol=SYMBOL,
    start_date=START_DATE,
    end_date=END_DATE,
    initial_capital=CAPITAL_INITIAL,
    max_allocation_pct=ALLOCATION_MAX,
    ratio_min=RATIO_MIN,
    ratio_max=RATIO_MAX,
    min_data_days=200,
    required_post_days=5
)

if results_df is not None:
    display_results(results_df, best_strategy, buy_hold_return)
else:
    print("Aucun résultat à afficher.")