"""
Descarga dataset Superintendencia Financiera Colombia
"""
import pandas as pd
from pathlib import Path

def download_sfc_data():
    url = "https://www.datos.gov.co/api/views/bn4c-kptq/rows.csv?accessType=DOWNLOAD"
    
    print("="*60)
    print("DESCARGANDO DATASET SFC COLOMBIA")
    print("="*60)
    print(f"URL: {url}")
    print("Esto tomará 2-3 minutos (~6M filas)...\n")
    
    try:
        df = pd.read_csv(url, low_memory=False)
        
        output_path = Path("data/raw")
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / "sfc_colombia_raw.csv"
        
        df.to_csv(output_file, index=False)
        
        print("✓ DESCARGA EXITOSA")
        print(f"  Filas: {df.shape[0]:,}")
        print(f"  Columnas: {df.shape[1]}")
        print(f"  Guardado: {output_file}\n")
        
        print("COLUMNAS:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        
        print("\nPRIMERAS 3 FILAS:")
        print(df.head(3))
        
        return df
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

if __name__ == "__main__":
    df = download_sfc_data()
