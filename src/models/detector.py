"""
Motor de inferencia para detección AML en tiempo real
Carga modelos entrenados y analiza transacciones individuales
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class AMLDetector(nn.Module):
    """Red neuronal para detección AML (arquitectura debe coincidir con train_and_export.py)"""
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class AMLPaymentDetector:
    """
    Detector de lavado de activos en pagos colombianos
    Usa ensemble XGBoost (65%) + PyTorch (35%)
    """
    
    def __init__(self, models_dir='models'):
        """Carga modelos entrenados"""
        self.models_dir = Path(models_dir)
        self._load_models()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_nn.to(self.device)
        self.model_nn.eval()
        
    def _load_models(self):
        """Carga todos los artefactos del entrenamiento"""
        print("🔄 Cargando modelos...")
        
        # XGBoost
        self.model_xgb = joblib.load(self.models_dir / "xgboost_aml.joblib")
        print("   ✓ XGBoost")
        
        # PyTorch
        checkpoint = torch.load(self.models_dir / "pytorch_aml.pth", 
                                map_location='cpu', weights_only=True)
        self.model_nn = AMLDetector(checkpoint['input_size'])
        self.model_nn.load_state_dict(checkpoint['model_state_dict'])
        print("   ✓ PyTorch")
        
        # Preprocessors
        self.scaler = joblib.load(self.models_dir / "scaler.joblib")
        self.label_encoder = joblib.load(self.models_dir / "label_encoder.joblib")
        print("   ✓ Scaler & LabelEncoder")
        
        # Metadata
        self.metadata = joblib.load(self.models_dir / "metadata.joblib")
        print("   ✓ Metadata")
        print(f"\n   Features: {self.metadata['n_features']}")
        print(f"   XGBoost AUC: {self.metadata['xgb_auc']:.4f}")
        print(f"   PyTorch AUC: {self.metadata['nn_auc']:.4f}\n")
        
    def _create_features(self, transaction: Dict) -> np.ndarray:
        """
        Crea las mismas features que en entrenamiento (SARLAFT)
        """
        # Extrae campos
        step = transaction.get('step', 1)
        tipo = transaction.get('type', 'TRANSFER')
        amount = transaction['amount']
        old_balance_orig = transaction['oldbalanceOrg']
        new_balance_orig = transaction.get('newbalanceOrig', max(0, old_balance_orig - amount))
        old_balance_dest = transaction['oldbalanceDest']
        new_balance_dest = transaction.get('newbalanceDest', old_balance_dest + amount)
        
        # Features básicas
        features = {
            'step': step,
            'amount': amount,
            'oldbalanceOrg': old_balance_orig,
            'newbalanceOrig': new_balance_orig,
            'oldbalanceDest': old_balance_dest,
            'newbalanceDest': new_balance_dest,
        }
        
        # Encode tipo
        features['type_encoded'] = self.label_encoder.transform([tipo])[0]
        
        # FEATURES SARLAFT (mismo orden que entrenamiento)
        features['amount_balance_ratio'] = amount / (old_balance_orig + 1)
        features['balance_change_orig'] = abs(old_balance_orig - new_balance_orig)
        features['balance_change_dest'] = abs(new_balance_dest - old_balance_dest)
        features['is_high_amount'] = int(amount > 10000000)
        features['is_very_high_amount'] = int(amount > 20000000)
        features['is_extreme_amount'] = int(amount > 30000000)
        features['is_round_amount'] = int((amount % 1000000 == 0) and (amount > 5000000))
        features['orig_balance_after_zero'] = int((new_balance_orig < 100) and (old_balance_orig > 1000000))
        features['dest_balance_was_zero'] = int(old_balance_dest < 100)
        features['orig_change_ratio'] = features['balance_change_orig'] / (old_balance_orig + 1)
        features['dest_change_ratio'] = features['balance_change_dest'] / (old_balance_dest + 1)
        features['amount_log'] = np.log1p(amount)
        features['balance_diff'] = old_balance_orig - old_balance_dest
        
        # Convierte a array en el orden correcto
        feature_vector = np.array([[features[name] for name in self.metadata['feature_names']]])
        return feature_vector
    
    def _get_suspicion_reasons(self, transaction: Dict, features: np.ndarray) -> List[str]:
        """Identifica razones específicas de sospecha"""
        reasons = []
        
        amount = transaction['amount']
        old_balance_orig = transaction['oldbalanceOrg']
        new_balance_orig = transaction.get('newbalanceOrig', max(0, old_balance_orig - amount))
        tipo = transaction.get('type', 'TRANSFER')
        
        # Analiza patrones
        if amount > 30000000:
            reasons.append(f"Monto extremadamente alto (${amount:,.0f} COP > $30M)")
        elif amount > 20000000:
            reasons.append(f"Monto muy alto (${amount:,.0f} COP > $20M)")
        elif amount > 10000000:
            reasons.append(f"Monto supera umbral SARLAFT (${amount:,.0f} COP > $10M)")
            
        if (amount % 1000000 == 0) and amount > 5000000:
            reasons.append(f"Monto redondo sospechoso (${amount/1000000:.0f}M - posible structuring)")
            
        if new_balance_orig < 100 and old_balance_orig > 1000000:
            reasons.append(f"Vacía completamente la cuenta origen (${old_balance_orig:,.0f} → $0)")
            
        ratio = amount / (old_balance_orig + 1)
        if ratio > 0.9:
            reasons.append(f"Retira >90% del balance disponible ({ratio*100:.1f}%)")
        elif ratio > 0.7:
            reasons.append(f"Retira >70% del balance disponible ({ratio*100:.1f}%)")
            
        if tipo == 'CASH_OUT' and amount > 15000000:
            reasons.append("Retiro en efectivo de monto alto (riesgo lavado)")
            
        if not reasons:
            reasons.append("Patrón de transacción dentro de rangos normales")
            
        return reasons
    
    def _get_recommended_action(self, prob: float, nivel_riesgo: str) -> str:
        """Determina acción recomendada según SARLAFT"""
        if prob < 0.30:
            return "APROBAR - Transacción de bajo riesgo"
        elif prob < 0.60:
            return "MONITOREAR - Registrar en sistema de alertas"
        elif prob < 0.85:
            return "RETENER - Requiere revisión por oficial de cumplimiento"
        else:
            return "BLOQUEAR + ROS - Reportar a UIAF (Unidad de Información y Análisis Financiero)"
    
    def analyze_payment(self, transaction: Dict) -> Dict:
        """
        Analiza una transacción individual
        
        Args:
            transaction: Dict con campos mínimos:
                - amount: float (monto en COP)
                - oldbalanceOrg: float (balance origen antes)
                - oldbalanceDest: float (balance destino antes)
                - type: str (TRANSFER, CASH_OUT, PAYMENT, DEBIT, CASH_IN)
                - step: int opcional (día)
                - newbalanceOrig: float opcional (calculado si no se provee)
                - newbalanceDest: float opcional (calculado si no se provee)
                
        Returns:
            Dict con veredicto completo
        """
        # Crea features
        features = self._create_features(transaction)
        features_scaled = self.scaler.transform(features)
        
        # Predicción XGBoost
        prob_xgb = self.model_xgb.predict_proba(features_scaled)[0, 1]
        
        # Predicción PyTorch
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            output = self.model_nn(features_tensor)
            prob_nn = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        # Ensemble (XGBoost 65% + PyTorch 35%)
        prob_ensemble = 0.65 * prob_xgb + 0.35 * prob_nn
        es_fraude = prob_ensemble > 0.5
        
        # Nivel de riesgo
        if prob_ensemble < 0.30:
            nivel_riesgo = "BAJO"
        elif prob_ensemble < 0.60:
            nivel_riesgo = "MEDIO"
        elif prob_ensemble < 0.85:
            nivel_riesgo = "ALTO"
        else:
            nivel_riesgo = "CRÍTICO"
        
        # Veredicto
        veredicto = "SOSPECHOSO ⚠️" if es_fraude else "LEGÍTIMO ✓"
        
        # Razones
        razones = self._get_suspicion_reasons(transaction, features)
        
        # Acción recomendada
        accion = self._get_recommended_action(prob_ensemble, nivel_riesgo)
        
        return {
            'veredicto': veredicto,
            'es_fraude': es_fraude,
            'probabilidad_fraude': prob_ensemble * 100,
            'nivel_riesgo': nivel_riesgo,
            'detalles_modelo': {
                'xgboost_prob': prob_xgb * 100,
                'pytorch_prob': prob_nn * 100,
                'ensemble_prob': prob_ensemble * 100,
                'peso_xgb': 65,
                'peso_pytorch': 35,
            },
            'razones_sospecha': razones,
            'accion_recomendada': accion,
            'transaccion': {
                'monto': transaction['amount'],
                'tipo': transaction.get('type', 'TRANSFER'),
                'balance_origen_antes': transaction['oldbalanceOrg'],
                'balance_origen_despues': transaction.get('newbalanceOrig', 
                    max(0, transaction['oldbalanceOrg'] - transaction['amount'])),
            }
        }


def print_result(result: Dict):
    """Imprime resultado de forma legible"""
    print("\n" + "="*80)
    print("📊 RESULTADO ANÁLISIS AML")
    print("="*80)
    
    # Veredicto principal
    print(f"\n🎯 VEREDICTO: {result['veredicto']}")
    print(f"   Probabilidad de fraude: {result['probabilidad_fraude']:.2f}%")
    print(f"   Nivel de riesgo: {result['nivel_riesgo']}")
    
    # Transacción
    print(f"\n💰 TRANSACCIÓN:")
    print(f"   Tipo: {result['transaccion']['tipo']}")
    print(f"   Monto: ${result['transaccion']['monto']:,.0f} COP")
    print(f"   Balance origen: ${result['transaccion']['balance_origen_antes']:,.0f} → ${result['transaccion']['balance_origen_despues']:,.0f}")
    
    # Modelos
    print(f"\n🤖 DETALLES MODELO:")
    print(f"   XGBoost (65%): {result['detalles_modelo']['xgboost_prob']:.2f}%")
    print(f"   PyTorch (35%): {result['detalles_modelo']['pytorch_prob']:.2f}%")
    print(f"   Ensemble:      {result['detalles_modelo']['ensemble_prob']:.2f}%")
    
    # Razones
    print(f"\n📋 RAZONES:")
    for i, razon in enumerate(result['razones_sospecha'], 1):
        print(f"   {i}. {razon}")
    
    # Acción
    print(f"\n⚡ ACCIÓN RECOMENDADA:")
    print(f"   {result['accion_recomendada']}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Test básico
    detector = AMLPaymentDetector()
    
    # Caso de prueba
    test_transaction = {
        'type': 'CASH_OUT',
        'amount': 35000000,  # 35M COP
        'oldbalanceOrg': 40000000,
        'oldbalanceDest': 5000000,
        'step': 100,
    }
    
    result = detector.analyze_payment(test_transaction)
    print_result(result)
