"""
Módulo de evaluación de modelos de clasificación.
Funciones reutilizables para métricas, matrices de confusión, curvas ROC, etc.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from typing import Dict, List, Optional, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calcula métricas completas de clasificación binaria.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones binarias
        y_pred_proba: Probabilidades predichas (opcional, para AUC-ROC)
        
    Returns:
        Diccionario con métricas: accuracy, precision, recall, f1, auc_roc, fpr
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # AUC-ROC solo si tenemos probabilidades
    if y_pred_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            metrics['auc_roc'] = 0.0
    
    # False Positive Rate
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str] = None):
    """
    Imprime reporte de clasificación formateado.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones binarias
        target_names: Nombres de las clases (default: ['Normal', 'Fraude'])
    """
    if target_names is None:
        target_names = ['Normal', 'Fraude']
    
    print("\n" + "="*60)
    print("REPORTE DE CLASIFICACIÓN")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=target_names))


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         title: str = "Matriz de Confusión",
                         ax: Optional[plt.Axes] = None,
                         class_names: List[str] = None) -> plt.Axes:
    """
    Grafica matriz de confusión.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones binarias
        title: Título del gráfico
        ax: Axes de matplotlib (opcional)
        class_names: Nombres de las clases
        
    Returns:
        Axes con el gráfico
    """
    if class_names is None:
        class_names = ['Normal', 'Fraude']
    
    cm = confusion_matrix(y_true, y_pred)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Verdadero')
    ax.set_xlabel('Predicho')
    
    return ax


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                   label: str = "Modelo",
                   ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Grafica curva ROC.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred_proba: Probabilidades predichas
        label: Etiqueta del modelo
        ax: Axes de matplotlib (opcional)
        
    Returns:
        Axes con el gráfico
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, label=f'{label} (AUC = {auc:.4f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Aleatorio (AUC = 0.5000)')
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
    ax.set_title('Curva ROC')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    return ax


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Crea tabla comparativa de múltiples modelos.
    
    Args:
        results: Diccionario de diccionarios con métricas por modelo
                 Ejemplo: {'XGBoost': {'accuracy': 0.95, ...}, 'RF': {...}}
                 
    Returns:
        DataFrame con comparación de modelos
    """
    df = pd.DataFrame(results).T
    
    # Ordena columnas
    metric_order = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'fpr']
    cols = [c for c in metric_order if c in df.columns]
    df = df[cols]
    
    # Formatea nombres de columnas
    df.columns = [c.replace('_', ' ').title() for c in df.columns]
    
    return df


def plot_model_comparison(results: Dict[str, Dict[str, float]], 
                         metrics: List[str] = None,
                         figsize: Tuple[int, int] = (12, 6)):
    """
    Grafica comparación visual de modelos.
    
    Args:
        results: Diccionario de diccionarios con métricas por modelo
        metrics: Lista de métricas a comparar (default: todas disponibles)
        figsize: Tamaño de la figura
    """
    df = compare_models(results)
    
    if metrics is not None:
        # Filtrar solo métricas solicitadas
        metrics_title = [m.replace('_', ' ').title() for m in metrics]
        df = df[[m for m in metrics_title if m in df.columns]]
    
    fig, ax = plt.subplots(figsize=figsize)
    df.plot(kind='bar', ax=ax)
    ax.set_title('Comparación de Modelos', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_xlabel('Modelo')
    ax.legend(title='Métricas', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig, ax
