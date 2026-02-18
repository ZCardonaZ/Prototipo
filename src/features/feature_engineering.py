"""
Ingeniería de features para detección de lavado de activos (AML).
Features derivadas relevantes para identificar patrones sospechosos.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any


def create_aml_features(df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Crea features derivadas para detección AML.
    
    Args:
        df: DataFrame con columnas base del dataset
        config: Configuración opcional con umbrales
        
    Returns:
        DataFrame con features adicionales
    """
    df = df.copy()
    
    # Configuración por defecto
    high_amount_threshold = 10_000_000  # 10M COP
    round_threshold = 100_000  # Montos múltiplos de 100k
    
    if config:
        high_amount_threshold = config.get('high_amount_threshold', high_amount_threshold)
        round_threshold = config.get('round_threshold', round_threshold)
    
    # 1. Ratio de cambio de balance origen
    df['balance_change_ratio_orig'] = np.where(
        df['oldbalanceOrg'] > 0,
        (df['oldbalanceOrg'] - df['newbalanceOrig']) / df['oldbalanceOrg'],
        0
    )
    
    # 2. Ratio de cambio de balance destino
    df['balance_change_ratio_dest'] = np.where(
        df['oldbalanceDest'] > 0,
        (df['newbalanceDest'] - df['oldbalanceDest']) / df['oldbalanceDest'],
        0
    )
    
    # 3. Ratio monto/balance origen (qué proporción del balance se mueve)
    df['amount_balance_ratio'] = np.where(
        df['oldbalanceOrg'] > 0,
        df['amount'] / df['oldbalanceOrg'],
        0
    )
    
    # 4. Flag de monto alto (sospechoso en AML)
    df['is_high_amount'] = (df['amount'] > high_amount_threshold).astype(int)
    
    # 5. Flag de monto "redondo" (múltiplo exacto - sospechoso)
    df['is_round_amount'] = (df['amount'] % round_threshold == 0).astype(int)
    
    # 6. Balance final origen cercano a cero (vaciado de cuenta)
    df['is_zero_balance_orig'] = (df['newbalanceOrig'] < 100).astype(int)
    
    # 7. Balance final destino cercano a cero (cuenta intermediaria)
    df['is_zero_balance_dest'] = (df['newbalanceDest'] < 100).astype(int)
    
    # 8. Ambos balances finales a cero (altamente sospechoso)
    df['both_balances_zero'] = (
        (df['newbalanceOrig'] < 100) & (df['newbalanceDest'] < 100)
    ).astype(int)
    
    # 9. Diferencia absoluta entre monto y cambio de balance (inconsistencias)
    expected_change_orig = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balance_inconsistency_orig'] = np.abs(df['amount'] - expected_change_orig)
    
    expected_change_dest = df['newbalanceDest'] - df['oldbalanceDest']
    df['balance_inconsistency_dest'] = np.abs(df['amount'] - expected_change_dest)
    
    # 10. Feature temporal: día de la semana del step (si step representa días)
    # Asumiendo que step es día del año (1-365)
    df['day_of_week'] = df['step'] % 7
    
    # 11. Feature temporal: es fin de semana (actividad inusual)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    print(f"✓ Features AML creadas: {len(df.columns) - len(df.columns)} nuevas columnas")
    print(f"  - Ratios de balance (3)")
    print(f"  - Flags de montos sospechosos (2)")
    print(f"  - Flags de balances (3)")
    print(f"  - Inconsistencias (2)")
    print(f"  - Features temporales (2)")
    
    return df


def get_feature_names(include_base: bool = True, 
                     include_derived: bool = True) -> list:
    """
    Retorna lista de nombres de features disponibles.
    
    Args:
        include_base: Incluir features base del dataset
        include_derived: Incluir features derivadas
        
    Returns:
        Lista de nombres de features
    """
    base_features = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest'
    ]
    
    derived_features = [
        'balance_change_ratio_orig', 'balance_change_ratio_dest',
        'amount_balance_ratio', 'is_high_amount', 'is_round_amount',
        'is_zero_balance_orig', 'is_zero_balance_dest', 'both_balances_zero',
        'balance_inconsistency_orig', 'balance_inconsistency_dest',
        'day_of_week', 'is_weekend'
    ]
    
    features = []
    if include_base:
        features.extend(base_features)
    if include_derived:
        features.extend(derived_features)
    
    return features


def add_encoded_type(df: pd.DataFrame, encoder=None):
    """
    Codifica columna 'type' de transacción.
    
    Args:
        df: DataFrame con columna 'type'
        encoder: LabelEncoder pre-entrenado (opcional)
        
    Returns:
        Tuple (df_encoded, encoder)
    """
    from sklearn.preprocessing import LabelEncoder
    
    df = df.copy()
    
    if encoder is None:
        encoder = LabelEncoder()
        df['type_encoded'] = encoder.fit_transform(df['type'])
    else:
        df['type_encoded'] = encoder.transform(df['type'])
    
    return df, encoder
