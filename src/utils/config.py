"""
Utilidades para cargar y gestionar configuración desde YAML.
"""
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Carga configuración desde archivo YAML.
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Diccionario con configuración
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Configuración cargada desde: {config_path}")
    return config


def get_param(config: Dict[str, Any], *keys, default=None) -> Any:
    """
    Obtiene parámetro anidado del config con valor por defecto.
    
    Args:
        config: Diccionario de configuración
        *keys: Claves anidadas para acceder al parámetro
        default: Valor por defecto si no existe
        
    Returns:
        Valor del parámetro o default
        
    Example:
        >>> config = load_config()
        >>> epochs = get_param(config, 'pytorch', 'epochs', default=100)
    """
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value
