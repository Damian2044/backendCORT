from __future__ import annotations

from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


MetodoClustering = Literal["datasets", "imagenes", "texto"]


class CrearSesionClusteringRequest(BaseModel):
    k: int = Field(..., gt=1)
    cardinalidades: list[Annotated[int, Field(gt=0)]] = Field(..., min_length=2)
    metodo: MetodoClustering
    escalar: bool = False

    @model_validator(mode="after")
    def validar_cardinalidades_vs_k(self) -> CrearSesionClusteringRequest:
        if len(self.cardinalidades) != self.k:
            raise ValueError("cardinalidades debe tener exactamente k valores.")
        return self


class AgregarPuntoTextoRequest(BaseModel):
    texto: str = Field(..., min_length=1)
    etiqueta_real: Optional[Union[int, str]] = None


class AgregarPuntoDatasetRequest(BaseModel):
    vector: List[float] = Field(..., min_length=1)
    etiqueta_real: Optional[Union[int, str]] = None


class ActualizarCardinalidadesRequest(BaseModel):
    cardinalidades: list[Annotated[int, Field(gt=0)]] = Field(..., min_length=1)
