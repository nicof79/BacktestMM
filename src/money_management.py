"""
money_management.py
-------------------
Module gérant la logique de capital, les ordres et la gestion FIFO des positions.
"""

def execute_trades(df, bull, bear, config):
    """
    Simule les ordres d’achat/vente en fonction des signaux.
    Renvoie le capital final et les positions exécutées.
    """
    capital = config["initial_capital"]
    allocation = config.get("max_allocation", 0.2)
    positions = []

    for i in range(1, len(df)):
        if bull.iloc[i]:  # Signal d'achat
            invest = capital * allocation
            qty = invest // df["Open"].iloc[i + 1] if i + 1 < len(df) else 0
            if qty > 0:
                capital -= qty * df["Open"].iloc[i + 1]
                positions.append(("BUY", df.index[i + 1], df["Open"].iloc[i + 1], qty))

        elif bear.iloc[i] and positions:  # Signal de vente
            side, date, price, qty = positions.pop(0)
            capital += qty * df["Open"].iloc[i + 1]

    return {"final_capital": capital, "positions": positions}
