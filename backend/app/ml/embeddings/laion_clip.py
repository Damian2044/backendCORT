from __future__ import annotations
from PIL import Image
from io import BytesIO
import numpy as np
import open_clip
import torch

class ExtractorLaionCLIP:
    """
    Extractor de embeddings con LAION-CLIP (OpenCLIP).
    """
    def __init__(self, 
                 usar_gpu: bool = True, 
                 modelo: str = "ViT-B-32",
                 normalizar: bool = True):
 


        self.modelo_nombre = modelo
        self.normalizar = normalizar

        if usar_gpu and torch.cuda.is_available():
            self.dispositivo = torch.device("cuda")
        else:
            self.dispositivo = torch.device("cpu")

        self.modelo, _, self.preprocesamiento = open_clip.create_model_and_transforms(self.modelo_nombre,pretrained="openai")
        self.modelo = self.modelo.to(self.dispositivo).eval()

        self.tokenizer = open_clip.get_tokenizer(self.modelo_nombre)

    def _to_pil(self, imagen) -> Image.Image:
        """Convierte una imagen dada como array NumPy (BGR o RGB), bytes o PIL Image a PIL Image RGB."""
        if isinstance(imagen, Image.Image):
            return imagen.convert("RGB")

        if isinstance(imagen, np.ndarray):

            if imagen.ndim not in (2, 3):
                raise ValueError("Array de imagen inválido")

            if imagen.ndim == 3:
                if imagen.shape[2] == 4:  # RGBA
                    imagen = imagen[:, :, :3]
                elif imagen.shape[2] == 3:  # BGR -> RGB
                    imagen = imagen[:, :, ::-1]

            if imagen.dtype != np.uint8:
                imagen = imagen.astype(np.uint8)

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

        return embedding.detach().to("cpu").float().numpy().astype(np.float32) # tensores en CPU por numpy


    def extraer_embedding_imagen(self, imagen) -> np.ndarray | None:
        """Extrae el embedding de una imagen dada"""

        try:
            imagen_pil = self._to_pil(imagen)
        except Exception as e:
            return None  # controlado: imagen mala/corrupta

        imagen_procesada = self.preprocesamiento(imagen_pil).unsqueeze(0).to(self.dispositivo)

        with torch.inference_mode():
            embedding = self.modelo.encode_image(imagen_procesada)

        return self._procesar_embedding(embedding)[0]
    
    def extraer_embedding_texto(self, texto: str) -> np.ndarray | None:
        """Extrae el embedding de un texto dado"""
        if texto is None:
            return None

        texto = texto.strip()
        if texto == "":
            return None  # controlado: texto vacío

        tokens = self.tokenizer([texto]).to(self.dispositivo)

        with torch.inference_mode():
            embedding = self.modelo.encode_text(tokens)
        
        return self._procesar_embedding(embedding)[0]

