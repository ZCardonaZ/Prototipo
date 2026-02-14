# 🛡️ Sistema de Detección de Lavado de Activos (AML) - Colombia

Sistema de detección de lavado de activos en pasarelas de pago colombianas, cumpliendo con la regulación SARLAFT (Sistema de Administración del Riesgo de Lavado de Activos y de la Financiación del Terrorismo).

## 📋 Descripción

Este proyecto es un prototipo de grado que utiliza Machine Learning para detectar transacciones sospechosas de lavado de activos en tiempo real. El sistema emplea un ensemble de XGBoost y PyTorch Neural Networks entrenados con features específicas para cumplimiento SARLAFT colombiano.

### ✨ Características Principales

- 🔍 **Detección en tiempo real** de transacciones sospechosas
- 🎯 **Ensemble de modelos** (XGBoost 65% + PyTorch 35%) con AUC ~0.89
- 📊 **13+ features SARLAFT** específicas para el contexto colombiano
- ⚖️ **Niveles de riesgo** (BAJO, MEDIO, ALTO, CRÍTICO)
- 📝 **Recomendaciones de acción** según regulación SARLAFT
- 🔄 **Interfaz CLI** con modo demo e interactivo

## 🛠️ Stack Tecnológico

- **Python 3.13**
- **Machine Learning**: XGBoost, PyTorch, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Serialización**: Joblib
- **GPU**: CUDA (RTX 4050) con fallback a CPU

## 📁 Estructura del Proyecto

```
Prototipo/
├── src/
│   ├── data/
│   │   ├── generate_synthetic.py    # Generación de dataset sintético
│   │   └── download_sfc.py          # Descarga datos SFC
│   ├── models/
│   │   ├── train_and_export.py      # Entrenamiento y exportación de modelos
│   │   └── detector.py              # Motor de inferencia
│   └── analyze_payment.py           # Script principal CLI
├── models/                          # Modelos entrenados (*.joblib, *.pth)
├── data/
│   └── synthetic/                   # Dataset sintético generado
├── notebooks/
│   └── 03_modelos/
│       └── 01_baseline_modelo.ipynb # Notebook experimental original
├── requirements.txt
└── README.md
```

## 🚀 Instalación

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/ZCardonaZ/Prototipo.git
   cd Prototipo
   ```

2. **Crear entorno virtual (recomendado)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # o
   venv\Scripts\activate     # Windows
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Uso

### Flujo Completo

```bash
# 1. Generar dataset sintético (100k transacciones)
python -m src.data.generate_synthetic

# 2. Entrenar y exportar modelos
python -m src.models.train_and_export

# 3. Analizar pagos (modo demo con 4 casos)
python -m src.analyze_payment

# 4. Modo interactivo (ingreso manual)
python -m src.analyze_payment --interactive
```

### Modo Demo

El modo demo ejecuta automáticamente 4 casos de prueba:

```bash
python -m src.analyze_payment
```

**Casos incluidos:**
1. ✅ Pago normal de supermercado (150K COP) → LEGÍTIMO
2. ⚠️ Transferencia con monto redondo (8M COP) → SOSPECHOSO
3. ⚠️ Cash-out que vacía cuenta (45M COP) → ALTO RIESGO
4. ✅ Compra pequeña en línea (89.9K COP) → LEGÍTIMO

### Modo Interactivo

Permite analizar transacciones personalizadas:

```bash
python -m src.analyze_payment --interactive
```

Ingresa los datos solicitados:
- Tipo de transacción (TRANSFER, CASH_OUT, PAYMENT, DEBIT, CASH_IN)
- Monto en COP
- Balance origen antes de la transacción
- Balance destino antes de la transacción
- Día del año (1-365)

### Uso Programático

```python
from src.models.detector import AMLPaymentDetector

# Inicializar detector
detector = AMLPaymentDetector()

# Crear transacción
transaction = {
    'type': 'CASH_OUT',
    'amount': 25000000,  # 25M COP
    'oldbalanceOrg': 30000000,
    'oldbalanceDest': 5000000,
    'step': 150,
}

# Analizar
result = detector.analyze_payment(transaction)

# Resultado incluye:
# - veredicto: "SOSPECHOSO ⚠️" o "LEGÍTIMO ✓"
# - probabilidad_fraude: float (0-100%)
# - nivel_riesgo: "BAJO", "MEDIO", "ALTO", "CRÍTICO"
# - razones_sospecha: list[str]
# - accion_recomendada: str (APROBAR, MONITOREAR, RETENER, BLOQUEAR+ROS)
# - detalles_modelo: dict con probabilidades de cada modelo
```

## 🔬 Detalles Técnicos

### Features SARLAFT (20 total)

El sistema genera 13 features adicionales específicas para SARLAFT:

1. `amount_balance_ratio` - Ratio monto vs balance origen
2. `balance_change_orig` - Cambio absoluto balance origen
3. `balance_change_dest` - Cambio absoluto balance destino
4. `is_high_amount` - Monto >10M COP (umbral SARLAFT)
5. `is_very_high_amount` - Monto >20M COP
6. `is_extreme_amount` - Monto >30M COP
7. `is_round_amount` - Montos redondos (posible structuring)
8. `orig_balance_after_zero` - Balance origen queda en ~0
9. `dest_balance_was_zero` - Cuenta destino dormida
10. `orig_change_ratio` - % del balance origen movido
11. `dest_change_ratio` - % del balance destino modificado
12. `amount_log` - Escala logarítmica del monto
13. `balance_diff` - Diferencia entre balances

### Modelos

- **XGBoost**: 200 estimadores, max_depth=6, scale_pos_weight balanceado
- **PyTorch NN**: 4 capas (128→64→32→1), Batch Normalization, Dropout 0.3
- **Ensemble**: 65% XGBoost + 35% PyTorch

### Mejoras Implementadas

✅ **Corregido bug double sigmoid** en PyTorch:
- Antes: `forward()` con sigmoid + `BCEWithLogitsLoss` → sigmoid doble
- Ahora: `forward()` sin sigmoid + `BCEWithLogitsLoss` correctamente
- Resultado: Recall mejoró de 67% a 99%

✅ **Balances negativos corregidos**:
- `newbalanceOrig = max(0, oldbalanceOrg - amount)`

✅ **Patrones de fraude realistas**:
- Structuring (montos redondos <10M)
- Smurfing (transferencias medianas múltiples)
- Layering (montos altos que vacían cuentas)
- No determinista (selección probabilística)

## 📊 Resultados

| Modelo | AUC-ROC | Precisión | Recall | F1-Score |
|--------|---------|-----------|--------|----------|
| XGBoost | 0.8931 | 0.04 | 0.51 | 0.08 |
| PyTorch | 0.8983 | 0.04 | 0.99 | 0.08 |
| Ensemble | ~0.89 | - | - | - |

*Nota: Baja precisión debido al desbalance extremo (1% fraude). El AUC-ROC es la métrica más relevante.*

## 📝 Regulación SARLAFT

El sistema implementa los siguientes umbrales según SARLAFT:

- **10M COP**: Umbral de reporte automático
- **20M COP**: Monto muy alto (alerta)
- **30M COP**: Monto extremo (bloqueo)

### Acciones Recomendadas

| Probabilidad | Nivel Riesgo | Acción |
|-------------|--------------|--------|
| <30% | BAJO | APROBAR - Transacción de bajo riesgo |
| 30-60% | MEDIO | MONITOREAR - Registrar en sistema de alertas |
| 60-85% | ALTO | RETENER - Revisión por oficial de cumplimiento |
| >85% | CRÍTICO | BLOQUEAR + ROS a UIAF |

## 🔧 Troubleshooting

### GPU no detectada
El sistema funciona en CPU automáticamente. Para usar GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Error al cargar modelos
Asegúrate de haber entrenado primero:
```bash
python -m src.models.train_and_export
```

### Dataset no encontrado
Genera el dataset sintético:
```bash
python -m src.data.generate_synthetic
```

## 🤝 Contribuciones

Este es un proyecto de grado en desarrollo. Sugerencias y mejoras son bienvenidas.

## 📄 Licencia

Este proyecto es un prototipo académico para la Universidad.

## 👨‍💻 Autor

**Proyecto de Grado** - Detección de Lavado de Activos en Pasarelas de Pago Colombianas

---

*Desarrollado con Python, XGBoost, PyTorch y cumplimiento SARLAFT*
