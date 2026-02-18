"""
Módulo de validación cruzada estratificada para modelos de clasificación.
Implementa K-Fold CV con SMOTE aplicado dentro de cada fold para evitar data leakage.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from typing import Dict, Any, List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')


def stratified_cv_with_smote(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    n_folds: int = 5,
    smote_params: Dict[str, Any] = None,
    random_state: int = 42,
    return_models: bool = False
) -> Dict[str, Any]:
    """
    Validación cruzada estratificada con SMOTE aplicado dentro de cada fold.
    
    IMPORTANTE: SMOTE se aplica SOLO en el conjunto de entrenamiento de cada fold
    para evitar data leakage. El conjunto de validación permanece intacto.
    
    Args:
        X: Features (numpy array)
        y: Labels (numpy array)
        model_fn: Función que retorna un modelo instanciado
        n_folds: Número de folds (default: 5)
        smote_params: Parámetros para SMOTE (default: None)
        random_state: Semilla aleatoria
        return_models: Si True, retorna los modelos entrenados
        
    Returns:
        Diccionario con métricas agregadas y por fold
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score
    )
    
    # Configuración SMOTE por defecto
    if smote_params is None:
        smote_params = {
            'sampling_strategy': 0.5,
            'k_neighbors': 5,
            'random_state': random_state
        }
    
    # Inicializa StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Almacena resultados por fold
    fold_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'auc_roc': []
    }
    
    trained_models = [] if return_models else None
    
    print(f"\n{'='*60}")
    print(f"VALIDACIÓN CRUZADA ESTRATIFICADA ({n_folds}-Fold)")
    print(f"{'='*60}")
    print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} features")
    print(f"Distribución de clases: {np.bincount(y)}")
    print(f"SMOTE: {smote_params}")
    print(f"{'='*60}\n")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # Split fold
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Aplica SMOTE SOLO en training (NO en validación)
        smote = SMOTE(**smote_params)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)
        
        print(f"Fold {fold}/{n_folds}:")
        print(f"  Train original: {X_train_fold.shape[0]} -> Tras SMOTE: {X_train_resampled.shape[0]}")
        print(f"  Distribución tras SMOTE: {np.bincount(y_train_resampled)}")
        print(f"  Validación (sin SMOTE): {X_val_fold.shape[0]}")
        
        # Entrena modelo
        model = model_fn()
        model.fit(X_train_resampled, y_train_resampled)
        
        # Predicciones en validación
        y_pred = model.predict(X_val_fold)
        
        # Probabilidades (si el modelo las soporta)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_val_fold)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_val_fold)
            else:
                y_proba = y_pred
        except:
            y_proba = y_pred
        
        # Calcula métricas
        fold_results['accuracy'].append(accuracy_score(y_val_fold, y_pred))
        fold_results['precision'].append(precision_score(y_val_fold, y_pred, zero_division=0))
        fold_results['recall'].append(recall_score(y_val_fold, y_pred, zero_division=0))
        fold_results['f1_score'].append(f1_score(y_val_fold, y_pred, zero_division=0))
        
        try:
            fold_results['auc_roc'].append(roc_auc_score(y_val_fold, y_proba))
        except:
            fold_results['auc_roc'].append(0.0)
        
        print(f"  Métricas: Acc={fold_results['accuracy'][-1]:.4f}, "
              f"Prec={fold_results['precision'][-1]:.4f}, "
              f"Rec={fold_results['recall'][-1]:.4f}, "
              f"F1={fold_results['f1_score'][-1]:.4f}, "
              f"AUC={fold_results['auc_roc'][-1]:.4f}\n")
        
        if return_models:
            trained_models.append(model)
    
    # Calcula estadísticas agregadas
    results = {
        'folds': fold_results,
        'mean': {k: np.mean(v) for k, v in fold_results.items()},
        'std': {k: np.std(v) for k, v in fold_results.items()},
        'n_folds': n_folds
    }
    
    if return_models:
        results['models'] = trained_models
    
    # Imprime resumen
    print(f"\n{'='*60}")
    print("RESUMEN DE VALIDACIÓN CRUZADA")
    print(f"{'='*60}")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
        mean = results['mean'][metric]
        std = results['std'][metric]
        print(f"{metric.upper():12s}: {mean:.4f} ± {std:.4f}")
    print(f"{'='*60}\n")
    
    return results


def compare_models_cv(
    X: np.ndarray,
    y: np.ndarray,
    models: Dict[str, Callable],
    n_folds: int = 5,
    smote_params: Dict[str, Any] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compara múltiples modelos usando validación cruzada estratificada.
    
    Args:
        X: Features
        y: Labels
        models: Diccionario {nombre_modelo: función_que_retorna_modelo}
        n_folds: Número de folds
        smote_params: Parámetros SMOTE
        random_state: Semilla aleatoria
        
    Returns:
        DataFrame con resultados comparativos
    """
    comparison_results = {}
    
    for model_name, model_fn in models.items():
        print(f"\n{'#'*60}")
        print(f"Evaluando: {model_name}")
        print(f"{'#'*60}")
        
        results = stratified_cv_with_smote(
            X, y, model_fn, n_folds, smote_params, random_state
        )
        
        # Guarda solo métricas agregadas
        comparison_results[model_name] = results['mean']
    
    # Convierte a DataFrame
    df_results = pd.DataFrame(comparison_results).T
    
    # Ordena por F1-score (métrica clave para fraude)
    df_results = df_results.sort_values('f1_score', ascending=False)
    
    return df_results
