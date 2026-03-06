from __future__ import annotations

from typing import Optional

import numpy as np
from river import preprocessing
from sklearn.decomposition import IncrementalPCA


class EscaladorOnline:
    """Escalador incremental por punto usando river.preprocessing.StandardScaler."""

    def __init__(self, habilitado: bool):
        self.habilitado = bool(habilitado)
        self._escalador = preprocessing.StandardScaler()

    @staticmethod
    def _vector_a_diccionario(punto: np.ndarray) -> dict[str, float]:
        """Convierte vector numpy a diccionario por indice de feature."""
        return {f"f_{indice}": float(valor) for indice, valor in enumerate(np.asarray(punto, dtype=float))}

    @staticmethod
    def _diccionario_a_vector(diccionario: dict[str, float], dimension: int) -> np.ndarray:
        """Reconstruye vector numpy desde diccionario escalado de river."""
        return np.asarray(
            [float(diccionario.get(f"f_{indice}", 0.0)) for indice in range(int(dimension))],
            dtype=float,
        )

    def transformar(self, punto: np.ndarray) -> tuple[np.ndarray, Optional[dict[str, float]]]:
        if not self.habilitado:
            return punto, None

        punto_np = np.asarray(punto, dtype=float).reshape(-1)
        punto_dict = self._vector_a_diccionario(punto_np)

        punto_escalado_dict = self._escalador.transform_one(punto_dict)
        self._escalador.learn_one(punto_dict)

        punto_escalado = self._diccionario_a_vector(punto_escalado_dict, punto_np.shape[0])
        return punto_escalado, punto_escalado_dict


class PCAOnline:
    """PCA incremental sin bloquear respuestas mientras arranca."""

    def __init__(self, habilitado: bool):
        self.habilitado = bool(habilitado)
        self._pca: Optional[IncrementalPCA] = None
        self._buffer_inicio: list[np.ndarray] = []
        self._buffer_actualizacion: list[np.ndarray] = []

    @property
    def listo(self) -> bool:
        return self._pca is not None

    def observar(self, punto: np.ndarray) -> dict:
        if not self.habilitado:
            return {"pca_habilitado": False, "pca_listo": False, "punto_pca": None}

        punto_2d = punto.reshape(1, -1)
        punto_pca: Optional[np.ndarray] = None

        if self._pca is None:
            self._buffer_inicio.append(punto.copy())
            if len(self._buffer_inicio) >= 2:
                lote_inicio = np.vstack(self._buffer_inicio)
                n_componentes = max(1, min(2, lote_inicio.shape[1], lote_inicio.shape[0]))
                self._pca = IncrementalPCA(n_components=n_componentes)
                self._pca.partial_fit(lote_inicio)
                self._buffer_inicio.clear()
                punto_pca = self._pca.transform(punto_2d)[0]
        else:
            punto_pca = self._pca.transform(punto_2d)[0]
            self._buffer_actualizacion.append(punto.copy())
            minimo_lote = max(2, int(getattr(self._pca, "n_components_", 2)))
            if len(self._buffer_actualizacion) >= minimo_lote:
                lote = np.vstack(self._buffer_actualizacion)
                self._pca.partial_fit(lote)
                self._buffer_actualizacion.clear()

        return {
            "pca_habilitado": True,
            "pca_listo": self.listo,
            "punto_pca": punto_pca,
        }

    def transformar_centroides(self, centroides: np.ndarray) -> Optional[np.ndarray]:
        if not self.habilitado or self._pca is None:
            return None
        if centroides.size == 0:
            return np.empty((0, int(getattr(self._pca, "n_components_", 0))), dtype=float)
        return self._pca.transform(np.asarray(centroides, dtype=float))
