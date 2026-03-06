from __future__ import annotations
from PIL import Image
from io import BytesIO
import numpy as np
import torch
from transformers import AutoProcessor, AutoModel


class ExtractorSiglip:
    """
    Extractor de embeddings con SigLIP2 (Transformers).
    - Imagen -> get_image_features
    - Texto  -> get_text_features
    """
    def __init__(
        self,
        usar_gpu: bool = True,
        modelo: str = "google/siglip2-base-patch16-256",
        normalizar: bool = True
    ):
        self.modelo_nombre = modelo
        self.normalizar = normalizar

        if usar_gpu and torch.cuda.is_available():
            self.dispositivo = torch.device("cuda")
        else:
            self.dispositivo = torch.device("cpu")

        self.procesador = AutoProcessor.from_pretrained(
            self.modelo_nombre,
            use_fast=True
        )

        self.modelo = AutoModel.from_pretrained(self.modelo_nombre)
        self.modelo = self.modelo.to(self.dispositivo).eval()

    def _to_pil(self, imagen) -> Image.Image:
        """Convierte una imagen dada como NumPy array, bytes o PIL Image a PIL Image RGB."""
        if isinstance(imagen, Image.Image):
            return imagen.convert("RGB")

        if isinstance(imagen, np.ndarray):
            if imagen.ndim not in (2, 3):
                raise ValueError("Array de imagen inválido.")

            if imagen.ndim == 3:
                if imagen.shape[2] == 4:
                    imagen = imagen[:, :, :3]
                elif imagen.shape[2] != 3:
                    raise ValueError("La imagen debe tener 3 o 4 canales.")

            if np.issubdtype(imagen.dtype, np.floating):
                minimo = float(imagen.min()) if imagen.size > 0 else 0.0
                maximo = float(imagen.max()) if imagen.size > 0 else 0.0

                if 0.0 <= minimo and maximo <= 1.0:
                    imagen = (imagen * 255).clip(0, 255).astype(np.uint8)
                else:
                    imagen = imagen.clip(0, 255).astype(np.uint8)

            elif imagen.dtype != np.uint8:
                imagen = imagen.clip(0, 255).astype(np.uint8)

            return Image.fromarray(imagen).convert("RGB")

        if isinstance(imagen, bytes):
            return Image.open(BytesIO(imagen)).convert("RGB")

        raise ValueError(f"Tipo no soportado: {type(imagen)}")

    def _procesar_embedding(self, embedding) -> np.ndarray:
        """
        - Normaliza L2 (opcional)
        - Convierte a numpy float32
        """
        if self.normalizar:
            embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        return embedding.detach().cpu().float().numpy().astype(np.float32)

    def extraer_embedding_imagen(self, imagen) -> np.ndarray | None:
        """Extrae el embedding de una imagen dada."""
        try:
            imagen_pil = self._to_pil(imagen)
            entradas = self.procesador(images=imagen_pil, return_tensors="pt")
            entradas = {k: v.to(self.dispositivo) for k, v in entradas.items()}

            with torch.inference_mode():
                embedding = self.modelo.get_image_features(**entradas)

            return self._procesar_embedding(embedding)[0]
        except Exception:
            return None

    def extraer_embedding_texto(self, texto: str) -> np.ndarray | None:
        """Extrae el embedding de un texto dado."""
        if texto is None:
            return None

        texto = texto.strip()
        if texto == "":
            return None

        try:
            entradas = self.procesador(
                text=[texto],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            entradas = {k: v.to(self.dispositivo) for k, v in entradas.items()}

            with torch.inference_mode():
                embedding = self.modelo.get_text_features(**entradas)

            return self._procesar_embedding(embedding)[0]
        except Exception:
            return None