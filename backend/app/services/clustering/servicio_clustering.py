from __future__ import annotations

from typing import Any

import numpy as np

from .servicio_sesiones import ServicioSesiones, serializar_numpy


class ServicioClustering:
    """Procesa puntos y operaciones del modelo CORT sobre sesiones existentes."""

    def __init__(self, servicio_sesiones: ServicioSesiones):
        self._servicio_sesiones = servicio_sesiones

    def _respuesta(self, success: bool, message: str, data: Any = None) -> dict:
        return {"success": bool(success), "message": str(message), "data": serializar_numpy(data)}

    @staticmethod
    def _validar_metodo_sesion(sesion, metodo_esperado: str) -> dict | None:
        if sesion.metodo != metodo_esperado:
            return {
                "success": False,
                "message": f"La sesion es de tipo `{sesion.metodo}` y este endpoint requiere `{metodo_esperado}`.",
                "data": {
                    "sesion_id": sesion.sesion_id,
                    "metodo_sesion": sesion.metodo,
                    "metodo_endpoint": metodo_esperado,
                },
            }
        return None

    def agregar_punto_dataset(
        self,
        sesion_id: str,
        *,
        vector: list[float] | np.ndarray,
        etiqueta_real: object = None,
    ) -> dict:
        sesion = self._servicio_sesiones.obtener_sesion(sesion_id)
        if sesion is None:
            return self._respuesta(False, "Sesion no encontrada.", {"sesion_id": sesion_id})

        error_metodo = self._validar_metodo_sesion(sesion, "datasets")
        if error_metodo is not None:
            return self._respuesta(**error_metodo)

        try:
            vector_np = np.asarray(vector, dtype=float).reshape(-1)
            if vector_np.size == 0:
                raise ValueError("vector vacio")
            respuesta = sesion.procesar_vector(vector_np, etiqueta_real=etiqueta_real)
            return self._respuesta(
                success=bool(respuesta.get("success", False)),
                message=str(respuesta.get("message", "")),
                data=respuesta.get("data"),
            )
        except Exception as error:
            return self._respuesta(
                False,
                "Vector de dataset invalido.",
                {"tipo_error": type(error).__name__, "detalle_error": str(error)},
            )

    def agregar_punto_texto(
        self,
        sesion_id: str,
        *,
        texto: str,
        etiqueta_real: object = None,
        extractor_embeddings: Any = None,
    ) -> dict:
        sesion = self._servicio_sesiones.obtener_sesion(sesion_id)
        if sesion is None:
            return self._respuesta(False, "Sesion no encontrada.", {"sesion_id": sesion_id})

        error_metodo = self._validar_metodo_sesion(sesion, "texto")
        if error_metodo is not None:
            return self._respuesta(**error_metodo)

        if extractor_embeddings is None:
            return self._respuesta(
                False,
                "No hay extractor de embeddings disponible en la aplicacion.",
                {"sesion_id": sesion_id, "metodo": "texto"},
            )

        embedding = extractor_embeddings.extraer_embedding_texto(texto)
        if embedding is None:
            return self._respuesta(
                False,
                "No se pudo extraer embedding de texto.",
                {"texto_recibido": texto},
            )

        vector_np = np.asarray(embedding, dtype=float).reshape(-1)
        respuesta = sesion.procesar_vector(vector_np, etiqueta_real=etiqueta_real)
        return self._respuesta(
            success=bool(respuesta.get("success", False)),
            message=str(respuesta.get("message", "")),
            data=respuesta.get("data"),
        )

    def agregar_punto_imagen(
        self,
        sesion_id: str,
        *,
        imagen_bytes: bytes,
        etiqueta_real: object = None,
        extractor_embeddings: Any = None,
    ) -> dict:
        sesion = self._servicio_sesiones.obtener_sesion(sesion_id)
        if sesion is None:
            return self._respuesta(False, "Sesion no encontrada.", {"sesion_id": sesion_id})

        error_metodo = self._validar_metodo_sesion(sesion, "imagenes")
        if error_metodo is not None:
            return self._respuesta(**error_metodo)

        if extractor_embeddings is None:
            return self._respuesta(
                False,
                "No hay extractor de embeddings disponible en la aplicacion.",
                {"sesion_id": sesion_id, "metodo": "imagenes"},
            )

        embedding = extractor_embeddings.extraer_embedding_imagen(imagen_bytes)
        if embedding is None:
            return self._respuesta(
                False,
                "No se pudo extraer embedding de imagen.",
                {"tamano_bytes": len(imagen_bytes)},
            )

        vector_np = np.asarray(embedding, dtype=float).reshape(-1)
        respuesta = sesion.procesar_vector(vector_np, etiqueta_real=etiqueta_real)
        return self._respuesta(
            success=bool(respuesta.get("success", False)),
            message=str(respuesta.get("message", "")),
            data=respuesta.get("data"),
        )

    def actualizar_cardinalidades(self, sesion_id: str, cardinalidades_nuevas: list[int] | np.ndarray) -> dict:
        sesion = self._servicio_sesiones.obtener_sesion(sesion_id)
        if sesion is None:
            return self._respuesta(False, "Sesion no encontrada.", {"sesion_id": sesion_id})

        respuesta = sesion.modelo.actualizar_tamanios_maximos(cardinalidades_nuevas)
        if bool(respuesta.get("success", False)):
            sesion.cardinalidades = np.asarray(sesion.modelo.cardinalidades, dtype=int).copy()
            data_limpia = {
                "cardinalidades": sesion.cardinalidades.copy(),
                "tamanios_actuales": sesion.modelo.tamanios_actuales.copy(),
            }
            return self._respuesta(
                success=True,
                message=str(respuesta.get("message", "")),
                data=data_limpia,
            )

        return self._respuesta(
            success=False,
            message=str(respuesta.get("message", "")),
            data=respuesta.get("data"),
        )
