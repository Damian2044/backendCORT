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
    def __init__(self, 
                 usar_gpu: bool = True, 
                 modelo: str = "google/siglip2-base-patch16-256",
                 normalizar: bool = True):
 


        self.modelo_nombre = modelo
        self.normalizar = normalizar

        if usar_gpu and torch.cuda.is_available():
            self.dispositivo = torch.device("cuda")
        else:
            self.dispositivo = torch.device("cpu")

        # Procesador (tokenizador + transformaciones de imagen)
        self.procesador = AutoProcessor.from_pretrained(self.modelo_nombre, use_fast=True)


        # Modelo
        self.modelo = AutoModel.from_pretrained(self.modelo_nombre)
        self.modelo = self.modelo.to(self.dispositivo).eval()



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


    def extraer_embedding_imagen(self, imagen) -> np.ndarray:
        """Extrae el embedding de una imagen dada"""

        try:
            imagen_pil = self._to_pil(imagen)
        except Exception as e:
            return None  # controlado: imagen mala/corrupta
        
        entradas = self.procesador(images=imagen_pil, return_tensors="pt")
        entradas = {k: v.to(self.dispositivo) for k, v in entradas.items()}


        with torch.inference_mode():
            embedding =  self.modelo.get_image_features(**entradas)

        return self._procesar_embedding(embedding)[0]
    
    def extraer_embedding_texto(self, texto: str) -> np.ndarray | None:
        """Extrae el embedding de un texto dado"""
        if texto is None:
            return None

        texto = texto.strip()
        if texto == "":
            return None  # controlado: texto vacío
        
        entradas = self.procesador(
            text=[texto], return_tensors="pt", padding=True, truncation=True
        )
        entradas = {k: v.to(self.dispositivo) for k, v in entradas.items()}
        with torch.inference_mode():
            embedding = self.modelo.get_text_features(**entradas)
        
        return self._procesar_embedding(embedding)[0]

#Prueba rápida
if __name__ == "__main__":
    #Medir el tiempo de extracción de texto
    
    extractor = ExtractorSiglip(modelo="google/siglip2-base-patch16-256", normalizar=True)
    import time
    inicio = time.time()
    texto = "Hola mi nombre es ChatGPT y estoy probando la extracción de embeddings con SigLIP2"
    embedding_texto = extractor.extraer_embedding_texto(texto)
    fin = time.time()
    #print("Embedding de texto:", embedding_texto)
    texto = "Hello my name is ChatGPT and I am testing embedding extraction with SigLIP2"
    embedding_texto2 = extractor.extraer_embedding_texto(texto)
    print("Embedding de texto 2:", embedding_texto2)
    print("Similitud coseno entre los textos:", np.dot(embedding_texto, embedding_texto2) / (np.linalg.norm(embedding_texto) * np.linalg.norm(embedding_texto2)))
    print(f"Tiempo de extracción de texto: {fin - inicio:.4f} segundos")
    # Para probar con una imagen, descomenta lo siguiente y asegúrate de tener una imagen válida
    # import cv2
    # imagen = cv2.imread("ruta/a/tu/imagen.jpg")
    # embedding_imagen = extractor.extraer_embedding_imagen(imagen)
    # print("Embedding de imagen:", embedding_imagen)
