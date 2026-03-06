from __future__ import annotations
from PIL import Image
from io import BytesIO
import numpy as np
import base64
import os
import requests

class ExtractorLaionCLIP:
    """
    Extractor de embeddings multimodal usando Jina CLIP API (gratuita).
    Misma interfaz que la versión CLIP original.
    """
    def __init__(self, 
                 usar_gpu: bool = True,  # ignorado, se mantiene por compatibilidad
                 modelo: str = "jina-clip-v2",
                 normalizar: bool = True):

        self.modelo_nombre = modelo
        self.normalizar = normalizar
        self.api_key = os.environ.get("JINA_API_KEY", "")
        self.url = "https://api.jina.ai/v1/embeddings"

    def _normalizar(self, vec: np.ndarray) -> np.ndarray:
        norma = np.linalg.norm(vec)
        if norma < 1e-12:
            return vec
        return (vec / norma).astype(np.float32)

    def _to_pil(self, imagen) -> Image.Image:
        if isinstance(imagen, Image.Image):
            return imagen.convert("RGB")
        if isinstance(imagen, np.ndarray):
            if imagen.ndim == 3:
                if imagen.shape[2] == 4:
                    imagen = imagen[:, :, :3]
                elif imagen.shape[2] == 3:
                    imagen = imagen[:, :, ::-1]
            if imagen.dtype != np.uint8:
                imagen = imagen.astype(np.uint8)
            return Image.fromarray(imagen).convert("RGB")
        if isinstance(imagen, bytes):
            return Image.open(BytesIO(imagen)).convert("RGB")
        raise ValueError(f"Tipo no soportado: {type(imagen)}")

    def _llamar_api(self, input_data: list) -> np.ndarray | None:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.post(
                self.url,
                headers=headers,
                json={"model": self.modelo_nombre, "input": input_data}
            )
            response.raise_for_status()
            vec = np.array(response.json()["data"][0]["embedding"], dtype=np.float32)
            if self.normalizar:
                vec = self._normalizar(vec)
            return vec
        except Exception as e:
            print(f">>> ERROR Jina API: {repr(e)}")
            return None

    def extraer_embedding_imagen(self, imagen) -> np.ndarray | None:
        """Extrae el embedding de una imagen usando Jina CLIP API."""
        try:
            imagen_pil = self._to_pil(imagen)
        except Exception as e:
            print(f">>> ERROR _to_pil: {repr(e)}")
            return None

        buffer = BytesIO()
        imagen_pil.save(buffer, format="JPEG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return self._llamar_api([{"image": b64}])

    def extraer_embedding_texto(self, texto: str) -> np.ndarray | None:
        """Extrae el embedding de un texto dado usando Jina CLIP API."""
        if texto is None:
            return None
        texto = texto.strip()
        if texto == "":
            return None

        return self._llamar_api([{"text": texto}])