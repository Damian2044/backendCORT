from __future__ import annotations

from io import BytesIO
import base64
import os

import numpy as np
import requests
from PIL import Image


class ExtractorJinaCLIP:
    """
    Extractor de embeddings multimodal usando Jina Embeddings API.
    """

    def __init__(
        self,
        modelo: str = "jina-clip-v2",
        normalizar: bool = True,
        timeout: int = 30,
    ) -> None:
        self.modelo_nombre = modelo
        self.normalizar = normalizar
        self.timeout = timeout

        # API key solo desde variable de entorno
        self.api_key = os.environ.get("JINA_API_KEY", "").strip()

        self.url = "https://api.jina.ai/v1/embeddings"

    def _to_pil(self, imagen) -> Image.Image:
        if isinstance(imagen, Image.Image):
            return imagen.convert("RGB")

        if isinstance(imagen, np.ndarray):
            if imagen.ndim != 3 or imagen.shape[2] not in (3, 4):
                raise ValueError("El ndarray debe tener forma HxWx3 o HxWx4.")

            if imagen.shape[2] == 4:
                imagen = imagen[:, :, :3]

            if np.issubdtype(imagen.dtype, np.floating):
                min_val = float(np.min(imagen)) if imagen.size > 0 else 0.0
                max_val = float(np.max(imagen)) if imagen.size > 0 else 0.0

                if 0.0 <= min_val and max_val <= 1.0:
                    imagen = (imagen * 255).clip(0, 255).astype(np.uint8)
                else:
                    imagen = imagen.clip(0, 255).astype(np.uint8)

            elif imagen.dtype != np.uint8:
                imagen = imagen.clip(0, 255).astype(np.uint8)

            return Image.fromarray(imagen).convert("RGB")

        if isinstance(imagen, bytes):
            return Image.open(BytesIO(imagen)).convert("RGB")

        raise ValueError(f"Tipo de imagen no soportado: {type(imagen)}")

    def _llamar_api(self, input_data: list) -> np.ndarray | None:
        if not self.api_key:
            print(">>> ERROR Jina API: falta la variable de entorno JINA_API_KEY")
            return None

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.modelo_nombre,
                "input": input_data,
                "normalized": self.normalizar,
                "embedding_type": "float",
            }

            response = requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

            data = response.json()
            vec = np.array(data["data"][0]["embedding"], dtype=np.float32)

            return vec

        except requests.HTTPError as e:
            detalle = ""
            try:
                detalle = response.text
            except Exception:
                pass

            print(f">>> ERROR HTTP Jina API: {e}. Detalle: {detalle}")
            return None

        except Exception as e:
            print(f">>> ERROR Jina API: {repr(e)}")
            return None

    def extraer_embedding_imagen(self, imagen) -> np.ndarray | None:
        try:
            imagen_pil = self._to_pil(imagen)

            buffer = BytesIO()
            imagen_pil.save(buffer, format="JPEG", quality=95)

            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return self._llamar_api([{"bytes": b64}])

        except Exception as e:
            print(f">>> ERROR procesando imagen: {repr(e)}")
            return None

    def extraer_embedding_texto(self, texto: str) -> np.ndarray | None:
        if texto is None:
            return None

        texto = texto.strip()
        if not texto:
            return None

        return self._llamar_api([texto])