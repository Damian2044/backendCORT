from fastapi import FastAPI
from app.api.clustering import router as clustering_router
from app.api.sesiones import router as sesiones_router
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.ml.embeddings.jina_clip_api import ExtractorJinaCLIP
#from app.ml.embeddings.laion_clip import ExtractorLaionCLIP
#from app.ml.embeddings.siglip import ExtractorSiglip
from app.services.clustering import ServicioClustering, ServicioSesiones
import traceback

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print(">>> STARTUP: cargando extractor...")
        app.state.extractor_embeddings = ExtractorJinaCLIP()
        app.state.servicio_sesiones = ServicioSesiones()
        app.state.servicio_clustering = ServicioClustering(app.state.servicio_sesiones)
        print(">>> STARTUP: extractor cargado OK")
    except Exception as e:
        print(">>> STARTUP FALLÓ:", repr(e))
        traceback.print_exc()
        raise  # importantísimo para que uvicorn muestre el error real
    yield
    print(">>> SHUTDOWN")


app = FastAPI(
    title="CORT API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de CORT"}


app.include_router(sesiones_router)
app.include_router(clustering_router)
