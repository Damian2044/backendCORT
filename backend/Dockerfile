# 1. Imagen base
FROM python:3.11-slim

# 2. Directorio dentro del contenedor
WORKDIR /app

# 3. Copiar dependencias
COPY requirements.txt .

# 4. Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el código del proyecto
COPY app ./app

# 6. Exponer puerto
EXPOSE 8000

# 7. Comando para iniciar la API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]