from fastapi import Request

def get_extractor(request: Request):
    return request.app.state.extractor_embeddings


def get_servicio_sesiones(request: Request):
    return request.app.state.servicio_sesiones


def get_servicio_clustering(request: Request):
    return request.app.state.servicio_clustering
