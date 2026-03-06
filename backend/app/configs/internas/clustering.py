"""
Configuracion interna para sesiones de clustering.

Aqui viven los flags y valores por defecto del pipeline online.
"""

# PCA incremental en sesiones de clustering.
habilitar_pca: bool = True

# Configuracion por defecto del modelo CORT.
metricas_aproximadas_por_defecto: bool = True
guardar_puntos_por_defecto: bool = False

# Tiempo maximo de inactividad permitido para una sesion.
ttl_sesion_segundos: int = 3600
