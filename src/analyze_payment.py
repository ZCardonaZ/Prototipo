"""
Script principal para análisis de pagos AML
Modos: demo (4 casos ejemplo) e interactivo
"""
import argparse
import sys
from pathlib import Path

# Asegura que src está en el path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.detector import AMLPaymentDetector, print_result


def demo_mode():
    """Ejecuta 4 casos de demostración"""
    print("\n" + "="*80)
    print("🚀 MODO DEMO - CASOS DE EJEMPLO")
    print("="*80)
    
    # Carga detector
    detector = AMLPaymentDetector()
    
    # Caso 1: Pago normal pequeño
    print("\n\n" + "🔷"*40)
    print("CASO 1: Pago normal de supermercado")
    print("🔷"*40)
    
    caso1 = {
        'type': 'PAYMENT',
        'amount': 150000,  # 150K COP
        'oldbalanceOrg': 5000000,
        'oldbalanceDest': 10000000,
        'step': 50,
    }
    result1 = detector.analyze_payment(caso1)
    print_result(result1)
    
    # Caso 2: Transferencia mediana sospechosa (monto redondo)
    print("\n\n" + "🔷"*40)
    print("CASO 2: Transferencia sospechosa (monto redondo)")
    print("🔷"*40)
    
    caso2 = {
        'type': 'TRANSFER',
        'amount': 8000000,  # 8M exacto (sospechoso)
        'oldbalanceOrg': 10000000,
        'oldbalanceDest': 500000,
        'step': 120,
    }
    result2 = detector.analyze_payment(caso2)
    print_result(result2)
    
    # Caso 3: Cash-out que vacía cuenta (MUY SOSPECHOSO)
    print("\n\n" + "🔷"*40)
    print("CASO 3: Cash-out que vacía completamente la cuenta")
    print("🔷"*40)
    
    caso3 = {
        'type': 'CASH_OUT',
        'amount': 45000000,  # 45M COP
        'oldbalanceOrg': 46000000,
        'oldbalanceDest': 2000000,
        'step': 200,
    }
    result3 = detector.analyze_payment(caso3)
    print_result(result3)
    
    # Caso 4: Compra pequeña normal
    print("\n\n" + "🔷"*40)
    print("CASO 4: Compra pequeña en línea")
    print("🔷"*40)
    
    caso4 = {
        'type': 'PAYMENT',
        'amount': 89900,  # 89.9K COP
        'oldbalanceOrg': 2500000,
        'oldbalanceDest': 8000000,
        'step': 180,
    }
    result4 = detector.analyze_payment(caso4)
    print_result(result4)
    
    # Resumen
    print("\n\n" + "="*80)
    print("📊 RESUMEN DE LOS 4 CASOS")
    print("="*80)
    
    casos = [
        ("Pago supermercado (150K)", result1),
        ("Transferencia redonda (8M)", result2),
        ("Cash-out vacía cuenta (45M)", result3),
        ("Compra en línea (89.9K)", result4),
    ]
    
    for i, (nombre, res) in enumerate(casos, 1):
        icono = "⚠️" if res['es_fraude'] else "✓"
        print(f"\n{i}. {nombre}")
        print(f"   {icono} {res['veredicto']} - Probabilidad: {res['probabilidad_fraude']:.1f}% - Riesgo: {res['nivel_riesgo']}")
        print(f"   Acción: {res['accion_recomendada']}")
    
    print("\n" + "="*80)


def interactive_mode():
    """Modo interactivo - usuario ingresa datos"""
    print("\n" + "="*80)
    print("🎮 MODO INTERACTIVO - ANÁLISIS DE TRANSACCIONES")
    print("="*80)
    
    # Carga detector
    detector = AMLPaymentDetector()
    
    print("\nIngrese los datos de la transacción:")
    print("(Presione Ctrl+C para salir)\n")
    
    while True:
        try:
            print("\n" + "-"*80)
            
            # Solicita datos
            tipo = input("Tipo de transacción (TRANSFER/CASH_OUT/PAYMENT/DEBIT/CASH_IN) [TRANSFER]: ").strip().upper()
            if not tipo:
                tipo = 'TRANSFER'
            if tipo not in ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'DEBIT', 'CASH_IN']:
                print("⚠️  Tipo inválido. Usando TRANSFER por defecto.")
                tipo = 'TRANSFER'
            
            amount_str = input("Monto en COP (ej: 5000000 para 5M): ").strip()
            if not amount_str:
                print("❌ Monto requerido.")
                continue
            try:
                amount = float(amount_str)
            except ValueError:
                print("❌ Monto inválido.")
                continue
            
            old_balance_orig_str = input("Balance origen antes (ej: 10000000 para 10M): ").strip()
            if not old_balance_orig_str:
                print("❌ Balance origen requerido.")
                continue
            try:
                old_balance_orig = float(old_balance_orig_str)
            except ValueError:
                print("❌ Balance origen inválido.")
                continue
            
            old_balance_dest_str = input("Balance destino antes (ej: 5000000 para 5M) [0]: ").strip()
            if not old_balance_dest_str:
                old_balance_dest = 0
            else:
                try:
                    old_balance_dest = float(old_balance_dest_str)
                except ValueError:
                    print("❌ Balance destino inválido. Usando 0.")
                    old_balance_dest = 0
            
            step_str = input("Día del año (1-365) [100]: ").strip()
            if not step_str:
                step = 100
            else:
                try:
                    step = int(step_str)
                except ValueError:
                    print("⚠️  Día inválido. Usando 100.")
                    step = 100
            
            # Crea transacción
            transaction = {
                'type': tipo,
                'amount': amount,
                'oldbalanceOrg': old_balance_orig,
                'oldbalanceDest': old_balance_dest,
                'step': step,
            }
            
            # Analiza
            result = detector.analyze_payment(transaction)
            print_result(result)
            
            # Continuar?
            continuar = input("\n¿Analizar otra transacción? (S/n): ").strip().lower()
            if continuar == 'n' or continuar == 'no':
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Saliendo...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Intente nuevamente.")
    
    print("\n" + "="*80)
    print("Gracias por usar el detector AML Colombia")
    print("="*80 + "\n")


def main():
    """Punto de entrada principal"""
    parser = argparse.ArgumentParser(
        description='Análisis de transacciones para detección de lavado de activos (SARLAFT Colombia)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python -m src.analyze_payment              # Modo demo (4 casos ejemplo)
  python -m src.analyze_payment -i           # Modo interactivo
  python -m src.analyze_payment --demo       # Modo demo explícito
        """
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Modo interactivo (usuario ingresa datos)'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Modo demo con casos predefinidos (por defecto)'
    )
    
    args = parser.parse_args()
    
    # Determina modo
    if args.interactive:
        interactive_mode()
    else:
        # Por defecto o --demo
        demo_mode()


if __name__ == "__main__":
    main()
