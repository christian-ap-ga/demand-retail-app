"""
Módulo de configuración central para el proyecto Demand_Retail.

Este módulo define las rutas base del proyecto a partir de la ubicación
del archivo y proporciona una función para inicializar un sistema de
logging estandarizado.

Funcionalidades principales:
- Resolución dinámica del directorio raíz del proyecto.
- Definición de rutas consistentes para datos crudos, procesados e inferencia.
- Creación automática de carpetas necesarias.
- Inicialización de logging con salida a archivo y consola, con nombre
  de proceso y timestamp.

Este módulo garantiza que todos los scripts (prep, train, inference)
usen la misma estructura de carpetas y el mismo formato de logs,
independientemente del directorio desde donde se ejecuten.
"""

import logging
import time
from datetime import datetime
from pathlib import Path

# Definición de rutas
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PREP_DIR = DATA_DIR / "prep"
INFERENCE_DIR = DATA_DIR / "inference"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
PREDICTIONS_DIR = DATA_DIR / "predictions"
LOG_DIR = ARTIFACTS_DIR / "logs"


def setup_logging(process_name: str):
    """
    Inicializa y configura el sistema de logging del proyecto.

    Crea las carpetas necesarias para artefactos y logs si no existen,
    y configura un logger que escribe tanto en archivo como en consola.

    El archivo de log se genera con un timestamp para evitar
    sobreescrituras entre ejecuciones.

    Parameters
    ----------
    process_name : str
        Nombre del proceso que se incluirá en el nombre del archivo
        de log y en el identificador del logger.

    Returns
    -------
    logging.Logger
        Logger configurado listo para ser utilizado en el pipeline.
    """
    for folder in [PREP_DIR, INFERENCE_DIR, LOG_DIR, ARTIFACTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / f"{process_name}_{timestamp}.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(process_name)


def log_process_time(logger: logging.Logger, start_time: float):
    """
    Registra en el log la finalización de un proceso y su tiempo total de ejecución.

    Calcula la duración transcurrida desde `start_time` hasta el momento actual
    y la escribe en el logger en segundos.

    Parameters
    ----------
    logger : logging.Logger
        Logger configurado donde se registrará la información.
    start_time : float
        Timestamp inicial del proceso (por ejemplo, obtenido con time.time()).

    Returns
    -------
    None
    """
    logger.info("Proceso terminado")
    duration = time.time() - start_time
    logger.info("Tiempo de ejecución: %.2f segundos", duration)
