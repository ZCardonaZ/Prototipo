# Prototipo v1 - Detección de Lavado de Activos en Pasarelas de Pago

## 📋 Descripción

Primer prototipo de la tesis de grado: **"Diseño de un modelo híbrido de aprendizaje automático para la detección de lavado de activos en pasarelas de pago"**.

Este proyecto implementa un sistema de detección de lavado de activos (AML - Anti-Money Laundering) utilizando técnicas de Machine Learning y Deep Learning sobre datos sintéticos de transacciones financieras en Colombia.

## 🎯 Objetivos Específicos

1. **OE1**: Caracterizar y analizar patrones de comportamiento asociados al lavado de activos
2. **OE2**: Implementar modelos de clasificación (XGBoost, Random Forest, Redes Neuronales)
3. **OE3**: Diseñar modelo híbrido con lógica difusa (futuro)
4. **OE4**: Evaluar y comparar modelos mediante validación cruzada
5. **OE5**: Implementar explicabilidad con SHAP y LIME

## 🏗️ Estructura del Proyecto

```
Prototipo/
├── configs/
│   └── config.yaml                 # Configuración centralizada
├── data/
│   ├── synthetic/                  # Datos sintéticos generados
│   ├── raw/                        # Datos crudos (futuro)
│   └── processed/                  # Datos procesados
├── notebooks/
│   ├── 01_eda/
│   │   └── 01_analisis_exploratorio.ipynb    # EDA completo
│   ├── 03_modelos/
│   │   └── 01_baseline_modelo.ipynb          # XGBoost + NN
│   └── 05_explicabilidad/
│       └── 01_shap_analysis.ipynb            # Explicabilidad SHAP
├── reports/
│   └── figures/                    # Gráficos generados
├── src/
│   ├── data/
│   │   ├── generate_synthetic.py   # Generación de datos sintéticos
│   │   └── download_sfc.py         # Descarga datos SFC Colombia
│   ├── features/
│   │   └── feature_engineering.py  # Ingeniería de features AML
│   ├── models/
│   │   ├── evaluation.py           # Métricas y evaluación
│   │   └── cross_validation.py     # Validación cruzada + SMOTE
│   ├── utils/
│   │   ├── config.py               # Carga configuración
│   │   └── reproducibility.py      # Semillas aleatorias
│   └── visualization/
├── requirements.txt                # Dependencias Python
└── README.md                       # Este archivo
```

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/ZCardonaZ/Prototipo.git
cd Prototipo
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 📊 Generación de Datos

Este proyecto utiliza **datos sintéticos** para cumplir con restricciones de privacidad y regulación.

### Generar dataset sintético

```bash
python src/data/generate_synthetic.py
```

Esto crea `data/synthetic/aml_colombia_synthetic.csv` con:
- 100,000 transacciones
- ~1% de fraude/lavado
- Tipos: TRANSFER, CASH_OUT, PAYMENT, DEBIT, CASH_IN
- Montos en COP (pesos colombianos)

## 🔬 Ejecución de Notebooks

### 1. Análisis Exploratorio de Datos (EDA)

```bash
jupyter notebook notebooks/01_eda/01_analisis_exploratorio.ipynb
```

Explora:
- Distribución de clases (fraude vs. normal)
- Patrones en montos y tipos de transacción
- Correlaciones entre variables
- Identificación de features discriminantes

### 2. Modelos Baseline

```bash
jupyter notebook notebooks/03_modelos/01_baseline_modelo.ipynb
```

Implementa:
- **XGBoost**: Gradient boosting con GPU
- **Red Neuronal Feedforward**: PyTorch con GPU
- Validación cruzada estratificada (5-fold)
- SMOTE para balanceo de clases
- Métricas: Precisión, Recall, F1-score, AUC-ROC

### 3. Explicabilidad (SHAP)

```bash
jupyter notebook notebooks/05_explicabilidad/01_shap_analysis.ipynb
```

Análisis de explicabilidad con SHAP values.

## 🛠️ Stack Tecnológico

- **Python 3.10+**
- **Machine Learning**: scikit-learn, XGBoost
- **Deep Learning**: PyTorch (GPU compatible)
- **Balanceo**: imbalanced-learn (SMOTE)
- **Explicabilidad**: SHAP, LIME
- **Visualización**: matplotlib, seaborn, plotly
- **Data**: pandas, numpy

## 📈 Pipeline de Detección

1. **Carga de datos** sintéticos/públicos
2. **Ingeniería de features** (ratios de balance, flags sospechosos)
3. **Preprocesamiento** (encoding, scaling)
4. **Balanceo de clases** (SMOTE dentro de CV)
5. **Entrenamiento de modelos** (XGBoost, RF, NN)
6. **Validación cruzada estratificada** (k=5)
7. **Evaluación** (métricas enfocadas en fraude)
8. **Explicabilidad** (SHAP, LIME)

## 🎯 Métricas Principales

Para detección de fraude (clase minoritaria):
- **Recall** (Sensibilidad): Detectar máximos fraudes posibles
- **F1-Score**: Balance entre Precisión y Recall
- **AUC-ROC**: Capacidad de discriminación
- **FPR**: Tasa de falsos positivos (minimizar)

## ⚖️ Contexto Regulatorio

- **Colombia**: SARLAFT 2.0 (Superintendencia Financiera)
- **Internacional**: GAFI (Grupo de Acción Financiera Internacional)
- **UIAF**: Unidad de Información y Análisis Financiero (Colombia)

**Nota**: Este es un prototipo académico con datos sintéticos. No implementa compliance engine real.

## 🔒 Consideraciones de Seguridad

- ✅ Solo datos sintéticos/públicos
- ✅ No se manejan datos reales de clientes
- ✅ Cumplimiento de privacidad por diseño
- ⚠️ **No usar en producción sin auditoría de seguridad**

## 📝 Reproducibilidad

El proyecto fija todas las semillas aleatorias para garantizar reproducibilidad:

```python
from src.utils.reproducibility import set_seed
set_seed(42)  # Fija numpy, torch, cuda, cudnn
```

## 🤝 Contribuciones

Este es un proyecto de tesis académica. Para contribuciones:

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -m 'Agrega nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es de código abierto para fines académicos.

## 👤 Autor

**ZCardonaZ**  
Tesis de Grado - Universidad [Nombre]  
Detección de Lavado de Activos con Machine Learning

## 📚 Referencias

- SARLAFT 2.0 (Superintendencia Financiera de Colombia)
- GAFI - Grupo de Acción Financiera Internacional
- Datasets sintéticos basados en PaySim/AMLSim

---

**Versión**: 1.0 (Prototipo)  
**Última actualización**: 2024
