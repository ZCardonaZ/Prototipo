"""
Script de entrenamiento mejorado AML Colombia
- Corrige bug double sigmoid PyTorch
- Feature engineering SARLAFT
- Exporta modelos entrenados
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Verifica GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AMLDetector(nn.Module):
    """Red neuronal para detección AML - SIN sigmoid en forward (usa BCEWithLogitsLoss)"""
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)  # Output sin sigmoid
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # NO SIGMOID - BCEWithLogitsLoss lo hace internamente
        return x


def create_sarlaft_features(df):
    """
    Crea features relevantes para detección SARLAFT Colombia
    """
    print("\n🔧 GENERANDO FEATURES SARLAFT...")
    
    # Features básicas existentes (7)
    features = df[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                   'oldbalanceDest', 'newbalanceDest']].copy()
    
    # Encode tipo de transacción
    le_type = LabelEncoder()
    features['type_encoded'] = le_type.fit_transform(df['type'])
    
    # FEATURES SARLAFT (13 adicionales)
    
    # 1. Ratio monto vs balance origen (riesgo si >0.8)
    features['amount_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    
    # 2. Cambio absoluto balance origen
    features['balance_change_orig'] = np.abs(df['oldbalanceOrg'] - df['newbalanceOrig'])
    
    # 3. Cambio absoluto balance destino
    features['balance_change_dest'] = np.abs(df['newbalanceDest'] - df['oldbalanceDest'])
    
    # 4. Monto alto (>10M COP - umbral SARLAFT)
    features['is_high_amount'] = (df['amount'] > 10000000).astype(int)
    
    # 5. Monto muy alto (>20M COP)
    features['is_very_high_amount'] = (df['amount'] > 20000000).astype(int)
    
    # 6. Monto extremo (>30M COP)
    features['is_extreme_amount'] = (df['amount'] > 30000000).astype(int)
    
    # 7. Monto redondo (posible structuring/smurfing)
    features['is_round_amount'] = ((df['amount'] % 1000000 == 0) & 
                                    (df['amount'] > 5000000)).astype(int)
    
    # 8. Balance origen queda en ~0 (vacía cuenta)
    features['orig_balance_after_zero'] = ((df['newbalanceOrig'] < 100) & 
                                             (df['oldbalanceOrg'] > 1000000)).astype(int)
    
    # 9. Balance destino estaba en ~0 (cuenta dormida)
    features['dest_balance_was_zero'] = (df['oldbalanceDest'] < 100).astype(int)
    
    # 10. Ratio de cambio origen (qué % del balance se movió)
    features['orig_change_ratio'] = features['balance_change_orig'] / (df['oldbalanceOrg'] + 1)
    
    # 11. Ratio de cambio destino
    features['dest_change_ratio'] = features['balance_change_dest'] / (df['oldbalanceDest'] + 1)
    
    # 12. Log del monto (escala logarítmica)
    features['amount_log'] = np.log1p(df['amount'])
    
    # 13. Diferencia balances (origen vs destino)
    features['balance_diff'] = df['oldbalanceOrg'] - df['oldbalanceDest']
    
    print(f"   ✓ Features totales: {features.shape[1]}")
    print(f"   ✓ Features SARLAFT: {features.shape[1] - 7}")
    
    return features, le_type


def train_models(data_path='data/synthetic/aml_colombia_synthetic.csv'):
    """
    Entrena modelos XGBoost y PyTorch con features SARLAFT
    """
    print("="*80)
    print("ENTRENAMIENTO MODELOS AML - SARLAFT COLOMBIA")
    print("="*80)
    
    # Carga datos
    print("\n📂 Cargando datos...")
    df = pd.read_csv(data_path)
    print(f"   Dataset: {df.shape}")
    print(f"   Fraude: {df['isFraud'].sum()} ({df['isFraud'].mean()*100:.2f}%)")
    
    # Crea features
    X, le_type = create_sarlaft_features(df)
    y = df['isFraud'].values
    
    # Split
    print("\n✂️  Train/Test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape[0]} samples ({y_train.sum()} fraudes)")
    print(f"   Test: {X_test.shape[0]} samples ({y_test.sum()} fraudes)")
    
    # Scale features
    print("\n🎚️  Escalando features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ========== XGBoost ==========
    print("\n" + "="*80)
    print("🌳 ENTRENANDO XGBOOST")
    print("="*80)
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"   Scale pos weight: {scale_pos_weight:.2f}")
    
    model_xgb = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='aucpr',
        tree_method='hist',
        device='cuda' if torch.cuda.is_available() else 'cpu'  # Requires XGBoost >= 2.0.0
    )
    
    model_xgb.fit(X_train_scaled, y_train)
    
    # Evalúa XGBoost
    y_pred_xgb = model_xgb.predict(X_test_scaled)
    y_proba_xgb = model_xgb.predict_proba(X_test_scaled)[:, 1]
    
    print("\n📊 RESULTADOS XGBOOST:")
    print(classification_report(y_test, y_pred_xgb, target_names=['Legítimo', 'Fraude']))
    print(f"   AUC-ROC: {roc_auc_score(y_test, y_proba_xgb):.4f}")
    
    # ========== PyTorch ==========
    print("\n" + "="*80)
    print("🔥 ENTRENANDO PYTORCH NEURAL NETWORK")
    print("="*80)
    print(f"   Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Prepara tensors
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)
    
    # Modelo
    model_nn = AMLDetector(X_train_t.shape[1]).to(device)
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.001)
    
    # Loss con weight (balanceo de clases)
    pos_weight = torch.tensor([scale_pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Incluye sigmoid internamente
    
    print(f"   Parámetros: {sum(p.numel() for p in model_nn.parameters()):,}")
    print(f"   BCEWithLogitsLoss (NO double sigmoid)")
    
    # Train
    epochs = 50
    batch_size = 512
    n_batches = len(X_train_t) // batch_size
    
    print(f"\n   Entrenando {epochs} epochs...")
    model_nn.train()
    for epoch in range(epochs):
        epoch_loss = 0
        indices = torch.randperm(len(X_train_t))
        
        for i in range(0, len(X_train_t), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_x = X_train_t[batch_idx]
            batch_y = y_train_t[batch_idx]
            
            optimizer.zero_grad()
            outputs = model_nn(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/n_batches:.4f}")
    
    # Evalúa PyTorch
    model_nn.eval()
    with torch.no_grad():
        outputs = model_nn(X_test_t)
        y_proba_nn = torch.sigmoid(outputs).cpu().numpy().flatten()  # Aplicar sigmoid AQUÍ para predicción
        y_pred_nn = (y_proba_nn > 0.5).astype(int)
    
    print("\n📊 RESULTADOS PYTORCH:")
    print(classification_report(y_test, y_pred_nn, target_names=['Legítimo', 'Fraude']))
    print(f"   AUC-ROC: {roc_auc_score(y_test, y_proba_nn):.4f}")
    
    # ========== Exporta modelos ==========
    print("\n" + "="*80)
    print("💾 EXPORTANDO MODELOS")
    print("="*80)
    
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    # XGBoost
    xgb_path = output_dir / "xgboost_aml.joblib"
    joblib.dump(model_xgb, xgb_path)
    print(f"   ✓ XGBoost: {xgb_path}")
    
    # PyTorch
    nn_path = output_dir / "pytorch_aml.pth"
    torch.save({
        'model_state_dict': model_nn.state_dict(),
        'input_size': X_train_t.shape[1],
    }, nn_path)
    print(f"   ✓ PyTorch: {nn_path}")
    
    # Scaler
    scaler_path = output_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"   ✓ Scaler: {scaler_path}")
    
    # Label Encoder
    le_path = output_dir / "label_encoder.joblib"
    joblib.dump(le_type, le_path)
    print(f"   ✓ LabelEncoder: {le_path}")
    
    # Metadata
    metadata = {
        'n_features': X_train.shape[1],
        'feature_names': list(X.columns),
        'xgb_auc': roc_auc_score(y_test, y_proba_xgb),
        'nn_auc': roc_auc_score(y_test, y_proba_nn),
        'scale_pos_weight': scale_pos_weight,
        'device_trained': str(device),
    }
    metadata_path = output_dir / "metadata.joblib"
    joblib.dump(metadata, metadata_path)
    print(f"   ✓ Metadata: {metadata_path}")
    
    print("\n" + "="*80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print(f"\nModelos exportados a: {output_dir.absolute()}")
    print(f"XGBoost AUC: {metadata['xgb_auc']:.4f}")
    print(f"PyTorch AUC: {metadata['nn_auc']:.4f}")
    
    return model_xgb, model_nn, scaler, le_type, metadata


if __name__ == "__main__":
    train_models()
