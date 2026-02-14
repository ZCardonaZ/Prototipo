"""
Genera dataset sintético lavado activos Colombia
100k transacciones - 1% fraude/lavado
"""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

def generate_aml_colombia():
    print("="*60)
    print("GENERANDO DATASET SINTÉTICO COLOMBIA")
    print("="*60)
    
    n_samples = 100000
    fraud_rate = 0.01  # 1% lavado
    
    # Tipos transacción Colombia
    tipos = ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'DEBIT', 'CASH_IN']
    
    # Genera datos
    data = {
        'step': np.random.randint(1, 365, n_samples),  # Días
        'type': np.random.choice(tipos, n_samples, p=[0.3, 0.25, 0.25, 0.1, 0.1]),
        'amount': np.random.lognormal(15, 2, n_samples),  # COP (lognormal realista)
        'nameOrig': ['C' + str(i) for i in np.random.randint(1, 5000, n_samples)],
        'oldbalanceOrg': np.random.uniform(0, 50000000, n_samples),
        'newbalanceOrig': 0,  # Calcular después
        'nameDest': ['M' + str(i) for i in np.random.randint(1, 5000, n_samples)],
        'oldbalanceDest': np.random.uniform(0, 50000000, n_samples),
        'newbalanceDest': 0,
    }
    
    df = pd.DataFrame(data)
    
    # Calcula balances
    df['newbalanceOrig'] = df['oldbalanceOrg'] - df['amount']
    df['newbalanceDest'] = df['oldbalanceDest'] + df['amount']
    
    # Genera fraude/lavado (patrones sospechosos)
    df['isFraud'] = 0
    df['isFlaggedFraud'] = 0
    
    # Patrones lavado:
    # 1. Montos muy altos (>20M COP)
    high_amount = df['amount'] > 20000000
    # 2. CASH_OUT frecuentes
    cash_out = df['type'] == 'CASH_OUT'
    # 3. Balances finales sospechosos (exactos a 0)
    zero_balance = (df['newbalanceOrig'] < 100) & (df['newbalanceDest'] < 100)
    
    # Marca fraudes (combinación patrones)
    fraud_mask = high_amount & (cash_out | zero_balance)
    fraud_indices = np.random.choice(
        df[fraud_mask].index, 
        size=min(int(n_samples * fraud_rate), fraud_mask.sum()),
        replace=False
    )
    df.loc[fraud_indices, 'isFraud'] = 1
    df.loc[df['amount'] > 30000000, 'isFlaggedFraud'] = 1
    
    # Guarda
    output_path = Path("data/synthetic")
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "aml_colombia_synthetic.csv"
    
    df.to_csv(output_file, index=False)
    
    print(f"✓ GENERADO")
    print(f"  Filas: {df.shape[0]:,}")
    print(f"  Fraude/Lavado: {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.2f}%)")
    print(f"  Monto promedio: ${df['amount'].mean():,.0f} COP")
    print(f"  Guardado: {output_file}\n")
    
    print("DISTRIBUCIÓN TIPOS:")
    print(df['type'].value_counts())
    
    print("\nESTADÍSTICAS FRAUDE:")
    print(df.groupby('isFraud')['amount'].describe())
    
    return df

if __name__ == "__main__":
    df = generate_aml_colombia()
