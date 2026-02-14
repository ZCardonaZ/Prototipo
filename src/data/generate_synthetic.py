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
    
    # Calcula balances (evita negativos)
    df['newbalanceOrig'] = np.maximum(0, df['oldbalanceOrg'] - df['amount'])
    df['newbalanceDest'] = df['oldbalanceDest'] + df['amount']
    
    # Genera fraude/lavado con patrones complejos y realistas
    df['isFraud'] = 0
    df['isFlaggedFraud'] = 0
    
    # Calcula indicadores auxiliares (sin etiquetarlos como fraude aún)
    high_amount = df['amount'] > 20000000  # >20M COP
    very_high_amount = df['amount'] > 30000000  # >30M COP
    round_amount = (df['amount'] % 1000000 == 0) & (df['amount'] > 5000000)  # Montos redondos (smurfing)
    cash_out = df['type'] == 'CASH_OUT'
    transfer = df['type'] == 'TRANSFER'
    
    # Balance patterns (sin usar como regla directa)
    depletes_account = (df['newbalanceOrig'] < 100) & (df['oldbalanceOrg'] > 1000000)  # Vacía cuenta
    high_ratio = (df['amount'] / (df['oldbalanceOrg'] + 1)) > 0.8  # >80% del balance
    
    # PATRONES DE LAVADO REALISTAS (múltiples combinaciones, no deterministas)
    
    # Patrón 1: Structuring - montos redondos frecuentes (evitan reportes >10M COP SARLAFT)
    structuring = round_amount & (df['amount'] < 10000000) & (cash_out | transfer)
    
    # Patrón 2: Smurfing - múltiples transferencias medianas
    smurfing = (df['amount'] > 3000000) & (df['amount'] < 9000000) & transfer
    
    # Patrón 3: Layering - montos altos que vacían cuentas
    layering = high_amount & depletes_account & cash_out
    
    # Patrón 4: Montos extremadamente altos con ratio sospechoso
    high_risk = very_high_amount & high_ratio
    
    # Combina patrones con probabilidades (NO determinista)
    fraud_candidates = structuring | smurfing | layering | high_risk
    
    # Selecciona aleatoriamente del conjunto de candidatos para llegar al 1%
    fraud_candidate_indices = df[fraud_candidates].index
    if len(fraud_candidate_indices) > 0:
        n_fraud = min(int(n_samples * fraud_rate), len(fraud_candidate_indices))
        fraud_indices = np.random.choice(fraud_candidate_indices, size=n_fraud, replace=False)
        df.loc[fraud_indices, 'isFraud'] = 1
    
    # isFlaggedFraud: sistema automático de alerta (>30M sin ser necesariamente fraude)
    df.loc[very_high_amount, 'isFlaggedFraud'] = 1
    
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
