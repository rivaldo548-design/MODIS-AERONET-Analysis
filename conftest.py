"""Configuración compartida de pytest para el repositorio."""

import sys
from pathlib import Path

# Permite ejecutar `pytest` desde cualquier ubicación asegurando que el
# paquete `scripts` (raíz del repositorio) sea importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
