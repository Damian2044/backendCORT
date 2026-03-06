from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_servicio_sesiones
from app.schemas.api_request import CrearSesionClusteringRequest
from app.schemas.api_response import ApiResponse

router = APIRouter(prefix="/sesiones", tags=["sesiones"])


@router.post("", response_model=ApiResponse[dict])
def crear_sesion(
    payload: CrearSesionClusteringRequest,
    servicio_sesiones=Depends(get_servicio_sesiones),
):
    resultado = servicio_sesiones.crear_sesion(
        k=payload.k,
        cardinalidades=payload.cardinalidades,
        metodo=payload.metodo,
        escalar=payload.escalar,
    )
    return ApiResponse(**resultado)


@router.get("", response_model=ApiResponse[dict])
def listar_sesiones(servicio_sesiones=Depends(get_servicio_sesiones)):
    resultado = servicio_sesiones.listar_sesiones()
    return ApiResponse(**resultado)


@router.get("/{sesion_id}", response_model=ApiResponse[dict])
def obtener_resumen_sesion(
    sesion_id: str,
    servicio_sesiones=Depends(get_servicio_sesiones),
):
    resultado = servicio_sesiones.obtener_resumen_sesion(sesion_id=sesion_id)
    return ApiResponse(**resultado)


@router.delete("/{sesion_id}", response_model=ApiResponse[dict])
def eliminar_sesion(
    sesion_id: str,
    servicio_sesiones=Depends(get_servicio_sesiones),
):
    resultado = servicio_sesiones.eliminar_sesion(sesion_id=sesion_id)
    return ApiResponse(**resultado)
