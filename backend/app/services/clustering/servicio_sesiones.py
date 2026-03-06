from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Dict, Optional
from uuid import uuid4

import numpy as np

from app.configs.internas.clustering import (
    guardar_puntos_por_defecto,
    habilitar_pca,
    metricas_aproximadas_por_defecto,
    ttl_sesion_segundos,
)
from app.ml.cort import CORTModelo
from .utilidades import EscaladorOnline, PCAOnline


def serializar_numpy(valor: Any) -> Any:
    """Convierte estructuras con numpy a tipos serializables JSON."""
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, dict):
        return {k: serializar_numpy(v) for k, v in valor.items()}
    if isinstance(valor, list):
        return [serializar_numpy(v) for v in valor]
    if isinstance(valor, tuple):
        return tuple(serializar_numpy(v) for v in valor)
    return valor


def marca_tiempo_monotonic() -> float:
    """Retorna un reloj monotonic para medir inactividad de sesiones."""
    return float(monotonic())


@dataclass
class SesionClustering:
    sesion_id: str
    metodo: str
    k: int
    cardinalidades: np.ndarray
    escalar: bool
    metricas_aproximadas: bool
    guardar_puntos: bool
    usar_pca: bool
    modelo: CORTModelo = field(init=False)
    escalador_online: EscaladorOnline = field(init=False)
    pca_online: PCAOnline = field(init=False)
    puntos_aceptados: int = 0
    puntos_rechazados: int = 0
    instante_creacion_monotonic: float = field(default_factory=marca_tiempo_monotonic)
    ultima_actividad_monotonic: float = field(init=False)

    def __post_init__(self) -> None:
        self.cardinalidades = np.asarray(self.cardinalidades, dtype=int).reshape(-1)
        self.ultima_actividad_monotonic = float(self.instante_creacion_monotonic)
        self.modelo = CORTModelo(
            k=self.k,
            cardinalidades=self.cardinalidades,
            metricas_aproximadas=self.metricas_aproximadas,
            guardar_puntos=self.guardar_puntos,
        )
        self.escalador_online = EscaladorOnline(habilitado=self.escalar)
        self.pca_online = PCAOnline(habilitado=self.usar_pca)

    def procesar_vector(self, vector: np.ndarray, etiqueta_real: object = None) -> dict:
        punto = np.asarray(vector, dtype=float).reshape(-1)

        punto_escalado, _ = self.escalador_online.transformar(punto)
        info_pca = self.pca_online.observar(punto_escalado)

        respuesta = self.modelo.asignar_punto(punto_escalado, etiqueta_real=etiqueta_real)
        if bool(respuesta.get("success", False)):
            self.puntos_aceptados += 1
        else:
            self.puntos_rechazados += 1

        data = dict(respuesta.get("data", {}))
        centroides_modelo = self.modelo.centroides_activos
        centroides_pca = self.pca_online.transformar_centroides(centroides_modelo)

        data["pca_habilitado"] = bool(info_pca.get("pca_habilitado", False))
        data["pca_listo"] = bool(info_pca.get("pca_listo", False))
        data["escalado_habilitado"] = self.escalador_online.habilitado
        data["punto_real"] = punto.copy()
        data["punto_pca"] = info_pca.get("punto_pca")
        data["centroides_modelo"] = centroides_modelo.copy()
        data["centroides_pca"] = centroides_pca
        return {
            "success": bool(respuesta.get("success", False)),
            "message": str(respuesta.get("message", "Punto procesado.")),
            "data": data,
        }

    def obtener_resumen(self) -> dict:
        respuesta = self.modelo.obtener_resumen_final(
            usar_aproximadas=self.metricas_aproximadas
        )
        data = dict(respuesta.get("data", {}))
        centroides_modelo = self.modelo.centroides_activos
        centroides_pca = self.pca_online.transformar_centroides(centroides_modelo)
        data["sesion_id"] = self.sesion_id
        data["metodo"] = self.metodo
        data["escalado_habilitado"] = self.escalador_online.habilitado
        data["pca_habilitado"] = self.pca_online.habilitado
        data["pca_listo"] = self.pca_online.listo
        data["centroides_modelo"] = centroides_modelo.copy()
        data["centroides_pca"] = centroides_pca
        return {
            "success": bool(respuesta.get("success", False)),
            "message": str(respuesta.get("message", "Resumen generado.")),
            "data": data,
        }


class ServicioSesiones:
    """Gestiona sesiones de clustering y su ciclo de vida."""

    _METODOS_VALIDOS = {"datasets": "datasets", "imagenes": "imagenes", "texto": "texto"}

    def __init__(self):
        self._sesiones: Dict[str, SesionClustering] = {}
        self._ttl_sesion_segundos = int(ttl_sesion_segundos)

    def _respuesta(self, success: bool, message: str, data: Any = None) -> dict:
        return {"success": bool(success), "message": str(message), "data": serializar_numpy(data)}

    @staticmethod
    def _ahora_monotonic() -> float:
        return float(monotonic())

    def _sesion_expirada(self, sesion: SesionClustering, ahora_monotonic: Optional[float] = None) -> bool:
        if self._ttl_sesion_segundos <= 0:
            return False
        ahora = self._ahora_monotonic() if ahora_monotonic is None else float(ahora_monotonic)
        return (ahora - float(sesion.ultima_actividad_monotonic)) >= self._ttl_sesion_segundos

    @staticmethod
    def _registrar_actividad_sesion(
        sesion: SesionClustering,
        ahora_monotonic: Optional[float] = None,
    ) -> None:
        sesion.ultima_actividad_monotonic = (
            marca_tiempo_monotonic() if ahora_monotonic is None else float(ahora_monotonic)
        )

    def _limpiar_sesiones_expiradas(self) -> None:
        if not self._sesiones:
            return

        ahora_monotonic = self._ahora_monotonic()
        sesiones_expiradas = [
            sesion_id
            for sesion_id, sesion in self._sesiones.items()
            if self._sesion_expirada(sesion, ahora_monotonic=ahora_monotonic)
        ]
        for sesion_id in sesiones_expiradas:
            self._sesiones.pop(sesion_id, None)

    def normalizar_metodo(self, metodo: str) -> Optional[str]:
        return self._METODOS_VALIDOS.get(str(metodo).strip().lower())

    def obtener_sesion(self, sesion_id: str) -> Optional[SesionClustering]:
        sesion_id = str(sesion_id)
        sesion = self._sesiones.get(sesion_id)
        if sesion is None:
            return None

        ahora_monotonic = self._ahora_monotonic()
        if self._sesion_expirada(sesion, ahora_monotonic=ahora_monotonic):
            self._sesiones.pop(sesion_id, None)
            return None

        self._registrar_actividad_sesion(sesion, ahora_monotonic=ahora_monotonic)
        return sesion

    def crear_sesion(
        self,
        *,
        k: int,
        cardinalidades: list[int] | np.ndarray,
        metodo: str,
        escalar: bool = False,
    ) -> dict:
        metodo_norm = self.normalizar_metodo(metodo)
        if metodo_norm is None:
            return self._respuesta(
                False,
                "Metodo no valido. Use datasets, imagenes o texto.",
                {"metodo_recibido": metodo, "metodos_validos": list(self._METODOS_VALIDOS.keys())},
            )

        try:
            self._limpiar_sesiones_expiradas()
            sesion_id = uuid4().hex
            sesion = SesionClustering(
                sesion_id=sesion_id,
                metodo=metodo_norm,
                k=int(k),
                cardinalidades=np.asarray(cardinalidades, dtype=int),
                escalar=bool(escalar),
                metricas_aproximadas=bool(metricas_aproximadas_por_defecto),
                guardar_puntos=bool(guardar_puntos_por_defecto),
                usar_pca=bool(habilitar_pca),
            )
            self._sesiones[sesion_id] = sesion
            return self._respuesta(
                True,
                "Sesion de clustering creada.",
                {
                    "sesion_id": sesion_id,
                    "k": sesion.k,
                    "cardinalidades": sesion.cardinalidades.copy(),
                    "metodo": sesion.metodo,
                    "escalado_habilitado": sesion.escalador_online.habilitado,
                    "pca_habilitado": sesion.pca_online.habilitado,
                    "pca_listo": sesion.pca_online.listo,
                    "ttl_sesion_segundos": self._ttl_sesion_segundos,
                },
            )
        except Exception as error:
            return self._respuesta(
                False,
                f"Error al crear sesion: {error}",
                {"tipo_error": type(error).__name__, "detalle_error": str(error)},
            )

    def obtener_resumen_sesion(self, sesion_id: str) -> dict:
        sesion = self.obtener_sesion(sesion_id)
        if sesion is None:
            return self._respuesta(False, "Sesion no encontrada.", {"sesion_id": sesion_id})

        respuesta = sesion.obtener_resumen()
        return self._respuesta(
            success=bool(respuesta.get("success", False)),
            message=str(respuesta.get("message", "")),
            data=respuesta.get("data"),
        )

    def eliminar_sesion(self, sesion_id: str) -> dict:
        sesion_id = str(sesion_id)
        sesion = self._sesiones.get(sesion_id)
        if sesion is None:
            return self._respuesta(False, "Sesion no encontrada.", {"sesion_id": sesion_id})

        if self._sesion_expirada(sesion):
            self._sesiones.pop(sesion_id, None)
            return self._respuesta(False, "Sesion no encontrada.", {"sesion_id": sesion_id})

        self._sesiones.pop(sesion_id, None)
        return self._respuesta(True, "Sesion eliminada.", {"sesion_id": sesion_id})

    def listar_sesiones(self) -> dict:
        self._limpiar_sesiones_expiradas()
        data = {
            "total_sesiones": len(self._sesiones),
            "sesiones": [
                {
                    "sesion_id": sesion.sesion_id,
                    "metodo": sesion.metodo,
                    "k": sesion.k,
                    "escalado_habilitado": sesion.escalador_online.habilitado,
                    "pca_habilitado": sesion.pca_online.habilitado,
                    "pca_listo": sesion.pca_online.listo,
                    "ttl_sesion_segundos": self._ttl_sesion_segundos,
                }
                for sesion in self._sesiones.values()
            ],
        }
        return self._respuesta(True, "Sesiones listadas.", data)
