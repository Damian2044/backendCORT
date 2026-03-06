from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.dependencies import get_extractor, get_servicio_clustering
from app.schemas.api_request import (
    ActualizarCardinalidadesRequest,
    AgregarPuntoDatasetRequest,
    AgregarPuntoTextoRequest,
)
from app.schemas.api_response import ApiResponse

router = APIRouter(prefix="/cort", tags=["clustering"])


def _normalizar_etiqueta_real(etiqueta_real: object) -> Optional[str]:
    if etiqueta_real is None:
        return None
    texto = str(etiqueta_real).strip()
    if texto == "":
        return None
    return texto


@router.post("/sesiones/{sesion_id}/puntos/datasets", response_model=ApiResponse[dict])
def agregar_punto_dataset(
    sesion_id: str,
    payload: AgregarPuntoDatasetRequest,
    servicio_clustering=Depends(get_servicio_clustering),
):
    resultado = servicio_clustering.agregar_punto_dataset(
        sesion_id=sesion_id,
        vector=payload.vector,
        etiqueta_real=_normalizar_etiqueta_real(payload.etiqueta_real),
    )
    return ApiResponse(**resultado)


@router.post("/sesiones/{sesion_id}/puntos/texto", response_model=ApiResponse[dict])
def agregar_punto_texto(
    sesion_id: str,
    payload: AgregarPuntoTextoRequest,
    servicio_clustering=Depends(get_servicio_clustering),
    extractor_embeddings=Depends(get_extractor),
):
    resultado = servicio_clustering.agregar_punto_texto(
        sesion_id=sesion_id,
        texto=payload.texto,
        etiqueta_real=_normalizar_etiqueta_real(payload.etiqueta_real),
        extractor_embeddings=extractor_embeddings,
    )
    return ApiResponse(**resultado)


@router.post("/sesiones/{sesion_id}/puntos/imagen", response_model=ApiResponse[dict])
async def agregar_punto_imagen(
    sesion_id: str,
    imagen: UploadFile = File(...),
    etiqueta_real: Optional[str] = Form(default=None),
    servicio_clustering=Depends(get_servicio_clustering),
    extractor_embeddings=Depends(get_extractor),
):
    imagen_bytes = await imagen.read()
    if not imagen_bytes:
        return ApiResponse(
            success=False,
            message="Imagen vacia.",
            data={"filename": imagen.filename},
        )

    resultado = servicio_clustering.agregar_punto_imagen(
        sesion_id=sesion_id,
        imagen_bytes=imagen_bytes,
        etiqueta_real=_normalizar_etiqueta_real(etiqueta_real),
        extractor_embeddings=extractor_embeddings,
    )
    return ApiResponse(**resultado)


@router.put("/sesiones/{sesion_id}/cardinalidades", response_model=ApiResponse[dict])
def actualizar_cardinalidades(
    sesion_id: str,
    payload: ActualizarCardinalidadesRequest,
    servicio_clustering=Depends(get_servicio_clustering),
):
    resultado = servicio_clustering.actualizar_cardinalidades(
        sesion_id=sesion_id,
        cardinalidades_nuevas=payload.cardinalidades,
    )
    return ApiResponse(**resultado)
