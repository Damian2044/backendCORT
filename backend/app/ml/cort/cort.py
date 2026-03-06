import numpy as np
class CORT:
    _EPSILON: float = 1e-12
    _C_SIGMA: float = 3.0
    _ETIQUETA_ERROR: int = -1

    def __init__(self, k: int, cardinalidades: list | np.ndarray):
        if len(cardinalidades) != k:
            raise ValueError(f"len(cardinalidades)={len(cardinalidades)} debe ser igual a k={k}")
        if any(e <= 0 for e in cardinalidades):
            raise ValueError("Todas las cardinalidades deben ser > 0")
        
        self.k = k
        self.cardinalidades = np.array(cardinalidades, dtype=int)
        self.totalPuntos:    int  = sum(cardinalidades)
        

        # Escala FFT online (máximo histórico de dt CAPADO)
        self._dmax_historico = 0.0


        # Estado de los clusters
        self.centroides: np.ndarray = np.zeros((k, 0), dtype=float)  # Se inicializa con 0 columnas, se ajustará al primer punto
        self.KFundados: int = 0
        self.cupoRestante    = np.array(cardinalidades, dtype=int)  # Cupo restante para cada cluster
        self.totalProcesados = 0


        # Welford sobre distancias máximas reales(Media y varianza para calcular cap_pre = mu_pre + 3*sigma_pre)
        self._dtContador: int = 0
        self._dtMedia: float = 0.0
        self._dtM2: float = 0.0

    @property
    def conteo(self) -> np.ndarray:# Conteo actual de puntos asignados a cada cluster, calculado como cardinalidades - cupoRestante
        return self.cardinalidades - self.cupoRestante
       
    # ─────────────────────────────────────────────────────────────────────────
    # WELFORD (dt REAL) + CAP PRE (mu + 3 sigma)
    # ─────────────────────────────────────────────────────────────────────────

    def _actualizar_media_m2_dt(self, dt_real: float):
        """Actualiza la media y M2 de Welford con el nuevo valor real de dt (distancia mínima máxima)"""
        self._dtContador += 1
        delta = dt_real - self._dtMedia #Diferencia entre el nuevo valor y la media actual
        self._dtMedia += delta / self._dtContador #Actualización de la media
        self._dtM2   += delta * (dt_real - self._dtMedia) #Actualización de M2, que es la suma de los cuadrados de las diferencias

    def _obtener_sigmaDt_pre(self) -> float:
        """Calcula el sigmaDt utilizando la desviación estándar de las distancias máximas reales, previas"""
        if self._dtContador < 2:
            return 0.0
        return float(np.sqrt(max(self._dtM2 / (self._dtContador - 1), 0.0)))
    
    def _capDt_pre(self) -> float:
        """Calcula el capDt_pre utilizando la media y el sigma de las distancias máximas reales previas"""
        # cap = mu_pre + 3*sigma_pre, calculado PRE (sin incluir dt actual)
        if self._dtContador < 2:
            return float("inf")
        sigma = self._obtener_sigmaDt_pre()
        return float(self._dtMedia + self._C_SIGMA * sigma)
    
    def _actualizar_escala_fft(self, dt_real: float):
        """
        Orden correcto:
          1) cap_pre = mu_pre + 3*sigma_pre   (histórico)
          2) dCap    = min(dt_real, cap_pre)
          3) DmaxGeom = max(DmaxGeom, dCap)
          4) Welford.update(dt_real)          (ahora sí incorporas el valor real)
        """
        if not np.isfinite(dt_real) or dt_real < 0:
            return

        cap_pre = self._capDt_pre() # Calcula el cap o límite previo basado en la media y desviación estándar de las distancias máximas reales anteriores
        dcap = min(dt_real, cap_pre)

        if dcap > self._dmax_historico:
            self._dmax_historico = dcap

        self._actualizar_media_m2_dt(dt_real) # Actualiza la media y M2 con el nuevo valor real de dt o distancia mínima máxima (capada)

    
    def _debe_fundarFFT(self, dMin_cupo: float, dmax_anterior: float) -> bool:
        t = self.totalProcesados
        N = self.totalPuntos
        umbral = (1.0 - np.sqrt((float(t) / float(N)))) * float(dmax_anterior)

        return float(dMin_cupo) > umbral

    # ─────────────────────────────────────────────────────────────────────────
    # UTILIDADES GEOMÉTRICAS
    # ─────────────────────────────────────────────────────────────────────────
    def _distancia_euclidiana(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))
    

    def _distancias_minimas(self, punto: np.ndarray) -> tuple:
        """
        Devuelve:
          dMin_cupo  : distancia al más cercano CON cupo  (decisión/asignación)
          jCercano   : índice del más cercano CON cupo (None si todos llenos)
          dMin_todos : distancia al más cercano SIN filtro (escala dt)
        """
        if self.KFundados <= 0:
            return float("inf"), None, float("inf")
        
        
        C = self.centroides[: self.KFundados] # Solo considera los centroides fundados hasta ahora
        diferencias = C - punto
        distancias = np.linalg.norm(diferencias, axis=1) # Distancia euclidiana a cada centroide fundado (sin filtro de cupo)


        # dMin de todos sin filtrar por cupo
        indice_min = np.argmin(distancias)
        dMin_todos = float(distancias[indice_min])

        # dMin al más cercano CON cupo
        mascara_cupo = self.cupoRestante[: self.KFundados] > 0
        if not np.any(mascara_cupo):
            return float("inf"), None, dMin_todos

        distancias_con_cupo = np.where(mascara_cupo, distancias, float("inf"))
        indice_min_cupo = np.argmin(distancias_con_cupo)
        dMin_cupo = float(distancias_con_cupo[indice_min_cupo])
        jCercano = int(indice_min_cupo)

        return dMin_cupo, jCercano, dMin_todos
   
    
    # ─────────────────────────────────────────────────────────────────────────
    # Actualización de centroides
    # ─────────────────────────────────────────────────────────────────────────
    def _actualizar_centroide(self, id: int, punto: np.ndarray):
        n_actual = int(self.cardinalidades[id] - self.cupoRestante[id])
        n_nuevo = n_actual + 1
        alpha   = 1.0 / (1.0 + np.sqrt(n_nuevo))
        self.centroides[id]   += alpha * (punto - self.centroides[id])
        # AQUÍ se actualiza el conteo indirectamente a través de cupoRestante, ya que conteo = cardinalidades - cupoRestante
        self.cupoRestante[id] -= 1

    # ─────────────────────────────────────────────────────────────────────────
    # FUNDAR CLUSTER/GRUPO
    # ─────────────────────────────────────────────────────────────────────────


    
    def _fundar_cluster(self, punto: np.ndarray) -> int:
        indice = self.KFundados
        if self.cupoRestante[indice] <= 0:
            return self._ETIQUETA_ERROR

        self.centroides[indice] = punto.copy()
        # Consumir un cupo del nuevo cluster fundado (equivalente a asignar el punto fundador a ese cluster)
        self.cupoRestante[indice] -= 1
        self.KFundados += 1
        

        # Cuando ya existen 2 centros, escala inicial = distancia entre ellos
        if self.KFundados == 2:
            d = float(self._distancia_euclidiana(self.centroides[0], self.centroides[1]))
            self._dmax_historico = d
            self._actualizar_media_m2_dt(d)

        return indice
    

    # ─────────────────────────────────────────────────────────────────────────
    # ASIGNACIÓN — con penalización por llenado
    # ─────────────────────────────────────────────────────────────────────────

    def _asignarConCapacidad(self, punto: np.ndarray) -> int:
        totalFundados = self.KFundados
        if totalFundados <= 0:
            return self._ETIQUETA_ERROR
        
        # Filtrar solo los clusters fundados y con cupo restante
        cupo = self.cupoRestante[:totalFundados]
        mascara_cupo = cupo > 0

        if not np.any(mascara_cupo):
            return self._ETIQUETA_ERROR
        
        centroides_activos = self.centroides[:totalFundados]
        distancias = np.linalg.norm(centroides_activos - punto, axis=1)

        # Si solo hay un cluster con cupo, asignar directamente a ese
        if np.count_nonzero(mascara_cupo) == 1:
            cluster = np.flatnonzero(mascara_cupo)[0] # Índice del único cluster con cupo
            self._actualizar_centroide(cluster, punto)
            return cluster


        # Análisis de candidatos con cupo: distancia + penalización por llenado
        distancias_candidatos = distancias[mascara_cupo]
        indices_candidatos = np.flatnonzero(mascara_cupo)   

        escala = max(float(np.median(distancias_candidatos)), self._EPSILON)


        # d1 y d2 para margen de ambigüedad, (si hay al menos 2 candidatos)
        dos_primeros = np.partition(distancias_candidatos, 1)[:2] # Obtiene las dos distancias más pequeñas entre los candidatos con cupo
        d1 = float(dos_primeros[0]) # Distancia al candidato más cercano con cupo
        d2 = float(dos_primeros[1]) # Distancia al segundo candidato con cupo

        ambiguedad = float(d1 / d2) if d2 > self._EPSILON else 1.0

        # penalización por llenado: cuanto más lleno, mayor penalización (función logarítmica suave)
        cardinalidades_candidatos =  np.asarray(self.cardinalidades, dtype=float)[:totalFundados][mascara_cupo]
        cupo_restante_candidatos = cupo[mascara_cupo].astype(float)

        llenado = (cardinalidades_candidatos - cupo_restante_candidatos) / cardinalidades_candidatos
        llenado = np.clip(llenado, 0.0, 1.0 - self._EPSILON)
        phi = - np.log(1.0 - llenado) # Penalización por llenado, función logarítmica suave que crece a medida que el llenado se acerca al 100%

        penalizacion = ambiguedad * escala * phi

        Ti = distancias_candidatos + penalizacion

        indice_menor_Ti = np.argmin(Ti)
        mejorGrupo = int(indices_candidatos[indice_menor_Ti])

        self._actualizar_centroide(mejorGrupo, punto)
        return mejorGrupo

    # ─────────────────────────────────────────────────────────────────────────
    # PROCESAMIENTO DE PUNTOS
    # ─────────────────────────────────────────────────────────────────────────


    def procesarPunto(self, punto) -> int:
        try:
            punto = np.asarray(punto, dtype=float)

            # inicializar centroides para que tengan la dimensión correcta al procesar el primer punto
            if self.centroides.shape[1] == 0:
                self.centroides = np.zeros((self.k, punto.shape[0]), dtype=float)



            etiqueta = self._ETIQUETA_ERROR

            # FASE 1: fundación (decidir fundar nuevo grupo o absorber en el más cercano con cupo)

            if self.KFundados < self.k:               
                if self.KFundados < 2: # p1 y p2 siempre fundan los 2 primeros clusters, luego se asigna con capacidad
                    etiqueta = self._fundar_cluster(punto)
                    if etiqueta != self._ETIQUETA_ERROR:
                        self.totalProcesados += 1
                    return etiqueta

                # desde p3: calcular distancias y actualizar escala FFT
                dMin_cupo, jCercano, dt = self._distancias_minimas(punto)
                dMax_anterior = self._dmax_historico # congelar Dmax PRE para decidir (esto es importante)


                # actualizar escala FFT online con dt (cada punto, funda o no)
                self._actualizar_escala_fft(dt)

                if jCercano is None: # si todos llenos → fundar obligatoriamente
                    etiqueta = self._fundar_cluster(punto)
                
                # decisión FFT con Dmax PRE (congelado antes de actualizar con dt actual)
                elif self._debe_fundarFFT(dMin_cupo, dMax_anterior):
                    etiqueta = self._fundar_cluster(punto)
                else:
                    etiqueta = self._asignarConCapacidad(punto)
            
             # FASE 2: asignación (score lagrangiano suave con penalización por llenado)
            else:
                etiqueta = self._asignarConCapacidad(punto)

            if etiqueta != self._ETIQUETA_ERROR:
                self.totalProcesados += 1

            return etiqueta          
        except Exception as e:
            print(f"Error al procesar el punto {punto}: {e}")
            return self._ETIQUETA_ERROR


    # ─────────────────────────────────────────────────────────────────────────
    # OTRAS UTILIDADES (actualizar los tamaños de los arrays si se cambia k o cardinalidades después de la inicialización)
    # ─────────────────────────────────────────────────────────────────────────
    def actualizar_tamanios_maximos(self, cardinalidades_nuevas):
        """
        Actualiza los tamaños máximos de los clusters.

        Reglas:
        - La longitud debe coincidir con k.
        - Ningún tamaño nuevo puede ser menor que el tamaño actual (conteo).
        - La actualización es atómica: si hay error no se cambia nada.
        """
        try:
            cardinalidades_nuevas = np.asarray(cardinalidades_nuevas, dtype=int)
            # validar longitud
            if cardinalidades_nuevas.ndim != 1 or cardinalidades_nuevas.shape[0] != self.k:
                return {
                    "success": False,
                    "message": f"Error: La longitud de cardinalidades_nuevas ({cardinalidades_nuevas.shape[0]}) no coincide con k ({self.k}).",
                    "data": None
                }
            
            # validar valores positivos
            if np.any(cardinalidades_nuevas <= 0):
                return {
                    "success": False,
                    "message": "Error: Todos los tamaños deben ser mayores que 0.",
                    "data": None
                }

            # cardinalidades máximas actuales 
            cardinalidades_actuales = self.cardinalidades
            # 1) Validar que los nuevos tamaños no sean menores que las cardinalidades actuales
            mascara_cardinalidades_iniciales = cardinalidades_nuevas < cardinalidades_actuales
            if np.any(mascara_cardinalidades_iniciales):
                idxs_problematicos = np.where(mascara_cardinalidades_iniciales)[0]
                detalles = [
                    f"Cluster {idx}: nuevo tamaño {cardinalidades_nuevas[idx]} < actual {cardinalidades_actuales[idx]}"
                    for idx in idxs_problematicos
                ]
                return {
                    "success": False,
                    "message": "Error: Los nuevos tamaños no pueden ser menores que las cardinalidades actuales.",
                    "data": detalles
                }

            # 2) Validar que los nuevos tamaños no sean menores que el conteo actual (puntos asignados)
            conteo_actual = self.conteo.copy()
            if np.any(cardinalidades_nuevas < conteo_actual):
                idxs_problematicos  = np.where(cardinalidades_nuevas < conteo_actual)[0]
                detalles = [
                    f"Cluster {i}: nuevo {int(cardinalidades_nuevas[i])} < asignados {int(conteo_actual[i])}"
                    for i in idxs_problematicos 
                ]
                return {
                    "success": False,
                    "message": "Error: Los nuevos tamaños no pueden ser menores que los puntos ya asignados.",
                    "data": detalles
                }



            # ───────────── aplicar cambios (atómico) ─────────────

            self.cardinalidades = cardinalidades_nuevas.copy()
            self.cupoRestante = (self.cardinalidades - conteo_actual).copy()
            self.totalPuntos = int(np.sum(self.cardinalidades))

            return {
                "success": True,
                "message": "Tamaños máximos actualizados exitosamente.",
                "data": {
                    "cardinalidades": self.cardinalidades.copy(),
                    "cupoRestante": self.cupoRestante.copy(),
                    "totalPuntos": self.totalPuntos
                }
            }

        except Exception as e:
            print(f"Error al actualizar tamaños máximos: {e}")
            return {
                "success": False,
                "message": f"Error inesperado: {e}",
                "data": None
            }




if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.utils import shuffle

    
    etiquetas = []

    data = load_iris()
    X = data.data
    y = data.target
    
    semillas=1
    inicial_seed=42
    silhouette_scores = []
    ari_scores = []
    
    for seed in range(inicial_seed, inicial_seed+semillas):
        etiquetas = []
        cort = CORT(k=3, cardinalidades=[50, 50, 50])
        X, y = shuffle(X, y, random_state=seed)
        for punto in X:
            etiqueta = cort.procesarPunto(punto)
            print(
                f"Punto: {punto}, Etiqueta asignada: {etiqueta}",
                f"Fundados: {cort.KFundados}, Cupo restante: {cort.cupoRestante}",
            )
            etiquetas.append(etiqueta)
        
        etiquetas = np.asarray(etiquetas)
        mask = (etiquetas != -1) 
            #if seed==1:
            #cort.actualizar_tamanios_maximos([100, 100, 100])
        try:
            sil_score = silhouette_score(X[mask], etiquetas[mask])
            ari_score = adjusted_rand_score(y[mask], etiquetas[mask])
            silhouette_scores.append(sil_score)
            ari_scores.append(ari_score)
        except Exception as e:
            continue


    
    print(f"\nResultados después de {semillas} ejecuciones con semillas {inicial_seed} a {inicial_seed+semillas-1}:")
    print(f"Silhouette Score: Promedio={np.mean(silhouette_scores):.4f}, Desviación estándar={np.std(silhouette_scores, ddof=1):.4f}")
    print(f"Adjusted Rand Index: Promedio={np.mean(ari_scores):.4f}, Desviación estándar={np.std(ari_scores, ddof=1):.4f}")
    