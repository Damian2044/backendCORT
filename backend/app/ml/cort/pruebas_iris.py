import sys
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import dataframe_image as dfi
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from pathlib import Path
from river import cluster as river_cluster
from river import preprocessing as river_preprocessing

# Agregar el directorio padre al path para importar el módulo de clustering
from .cort_modelo import CORTModelo


def prepararDirectorioResultados():
    """
    Prepara el directorio de resultados, borrándolo si existe.
    
    Args:
        None
    
    Returns:
        Path: Ruta del directorio de resultados creado.
    """
    dirResultados = Path(__file__).parent.parent / "resultados_iris"
    
    if dirResultados.exists():
        shutil.rmtree(dirResultados)
        print(f"✓ Directorio existente eliminado: {dirResultados}")
    
    dirResultados.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directorio creado: {dirResultados}")
    
    return dirResultados


def cargarDatasetIris():
    """
    Carga el dataset Iris desde sklearn.
    
    Args:
        None
    
    Returns:
        tuple: (datosOriginales: ndarray [150x4], etiquetasOriginales: ndarray de strings ['setosa', 'versicolor', 'virginica'])
    """
    iris = load_iris()
    datosOriginales = iris.data
    
    # Mapear etiquetas numéricas a nombres directamente
    mapeoNombres = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    etiquetasOriginales = np.array([mapeoNombres[int(e)] for e in iris.target])
    
    print(f"\n{'='*100}")
    print(f"DATASET IRIS CARGADO")
    print(f"{'='*100}")
    print(f"Muestras: {len(datosOriginales)}")
    print(f"Características: {datosOriginales.shape[1]}")
    print(f"Clases: {len(np.unique(etiquetasOriginales))}")
    
    return datosOriginales, etiquetasOriginales


def ejecutarClusteringConSemilla(datosOriginales, etiquetasOriginales, semilla):
    """
    Ejecuta clustering CluORT (basado en restricciones) con una semilla específica.
    
    Args:
        datosOriginales (ndarray): Datos de entrada [n_samples x n_features]
        etiquetasOriginales (ndarray): Etiquetas reales de clase [n_samples]
        semilla (int): Semilla para reproducibilidad de randomización
    
    Returns:
        dict: Diccionario con métricas:
            - silueta (float): Coeficiente de silueta
            - dunn (float): Índice de Dunn
            - ari (float): Adjusted Rand Index
            - ami (float): Adjusted Mutual Information
            - nmi (float): Normalized Mutual Information
            - distribucion (dict): Distribución de etiquetas reales por cluster
            - tamaniosActuales (list): Tamaños actuales de los clusters
            - tamaniosMaximos (list): Tamaños máximos permitidos
            - primeras5Etiquetas (str): Primeras 5 etiquetas reales
    """
    # Randomizar datos con la semilla
    datosShuffled, etiquetasShuffled = shuffle(datosOriginales, etiquetasOriginales, random_state=semilla)
    
    # Calcular número de clusters desde etiquetas únicas
    etiquetasUnicas, tamaniosMaximos = np.unique(etiquetasShuffled, return_counts=True)
    numClusters = len(etiquetasUnicas)
    tamaniosMaximos = tamaniosMaximos.tolist()
    
    # Inicializar clustering
    clustering = CORTModelo(k=numClusters, 
                            cardinalidades=tamaniosMaximos,
                            guardar_puntos=False,
                            metricas_aproximadas=True)
    
    # Procesar todos los puntos
    ultimoResultado = None
    for punto, etiqueta in zip(datosShuffled, etiquetasShuffled):
        ultimoResultado = clustering.asignar_punto(punto, etiqueta_real=etiqueta)
    
    ultimoResultado = ultimoResultado['data']
    print(f"  CluORT ejecutado con semilla {semilla}. Último punto procesado: {ultimoResultado}")
    # Extraer métricas finales
    metricasInternas = ultimoResultado['metricas_internas']
    metricasExternas = ultimoResultado['metricas_externas']
    distribucion = ultimoResultado['distribucion']
    
    # Obtener primeras 5 etiquetas reales (ya son nombres)
    primeras5EtiquetasStr = ', '.join(etiquetasShuffled[:5])
    
    return {
        'silueta': metricasInternas['silueta'],
        'dunn': metricasInternas['dunn'],
        'ari': metricasExternas['ari'],
        'ami': metricasExternas['ami'],
        'nmi': metricasExternas['nmi'],
        'distribucion': distribucion,
        'tamaniosActuales': clustering.tamanios_actuales.copy(),
        'tamaniosMaximos': clustering.cardinalidades.copy(),
        'primeras5Etiquetas': primeras5EtiquetasStr
    }

def ejecutarSTREAMKMeansConSemilla(datosOriginales, etiquetasOriginales, semilla):
    """
    Ejecuta STREAMKMeans (river) en modo verdaderamente ONLINE (streaming real punto a punto).
    Escalado en vivo: cada punto se escala y aprende independientemente.
    
    Args:
        datosOriginales (ndarray): Datos de entrada [n_samples x n_features]
        etiquetasOriginales (ndarray): Etiquetas reales de clase [n_samples]
        semilla (int): Semilla para reproducibilidad del algoritmo
    
    Returns:
        dict: Diccionario con métricas:
            - silueta (float): Coeficiente de silueta
            - dunn (float): Índice de Dunn (separación min / diámetro máx)
            - ari (float): Adjusted Rand Index
            - ami (float): Adjusted Mutual Information
            - nmi (float): Normalized Mutual Information
            - distribucion (dict): Distribución de etiquetas reales por cluster
            - tamaniosActuales (list): Tamaños actuales de los clusters
            - tamaniosMaximos (list): Tamaños máximos de las clases
            - primeras5Etiquetas (str): Primeras 5 etiquetas reales
    """
    # Randomizar datos
    datosShuffled, etiquetasShuffled = shuffle(
        datosOriginales, etiquetasOriginales, random_state=semilla
    )

    # Número de clusters
    etiquetasUnicas, tamaniosMaximos = np.unique(etiquetasShuffled, return_counts=True)
    numClusters = len(etiquetasUnicas)
    tamaniosMaximos = tamaniosMaximos.tolist()

    # Escalador ONLINE de river (streaming)
    scaler = river_preprocessing.StandardScaler()

    # STREAMKMeans ONLINE de river
    streamKMeans = river_cluster.STREAMKMeans(
        n_clusters=numClusters,
        seed=semilla,
        sigma=0.1
    )

    datosEscalados = []
    etiquetasPredichas = []
    
    # Procesar todos los puntos: escalar EN VIVO + predecir + aprender
    for i in range(len(datosShuffled)):
        x = datosShuffled[i]
        # Diccionario con los valores originales
        x_dict = {j: float(x[j]) for j in range(len(x))}
        
        # Actualizar escalador con el nuevo punto y obtener el punto escalado
        scaler.learn_one(x_dict)  # actualiza el escalador con el nuevo punto
        xEscalado = scaler.transform_one(x_dict)  # devuelve el punto escalado
        
        # Primeros k puntos: solo aprender para inicializar centros
        if i < numClusters:
            cluster = 0  # Placeholder
            streamKMeans.learn_one(xEscalado)
        else:
            # Resto: predecir y aprender
            cluster = streamKMeans.predict_one(xEscalado) if streamKMeans.centers else 0
            streamKMeans.learn_one(xEscalado)
        
        etiquetasPredichas.append(cluster)
        datosEscalados.append(np.array([xEscalado[j] for j in range(len(xEscalado))]))
    
    # Asignar predicciones correctas para los primeros k puntos usando el modelo entrenado
    datosEscalados_temp = np.array(datosEscalados)
    etiquetasPredichas_temp = np.array(etiquetasPredichas)
    
    for i in range(min(numClusters, len(datosShuffled))):
        x = datosShuffled[i]
        x_dict = {j: float(x[j]) for j in range(len(x))}
        # Solo transformar (el escalador ya aprendió estos puntos en la iteración anterior)
        xEscalado = scaler.transform_one(x_dict)
        if streamKMeans.centers:
            etiquetasPredichas_temp[i] = streamKMeans.predict_one(xEscalado)
    
    datosEscalados = datosEscalados_temp
    etiquetasPredichas = etiquetasPredichas_temp

    datosEscalados = np.array(datosEscalados)
    etiquetasPredichas = np.array(etiquetasPredichas)

    # Métricas internas
    silueta = silhouette_score(datosEscalados, etiquetasPredichas)

    # Índice de Dunn: separación mínima / diámetro máximo
    separacionMin = float('inf')
    for i in range(numClusters):
        puntos_i = datosEscalados[etiquetasPredichas == i]
        if len(puntos_i) == 0:
            continue
        for j in range(i + 1, numClusters):
            puntos_j = datosEscalados[etiquetasPredichas == j]
            if len(puntos_j) == 0:
                continue
            for punto_i in puntos_i:
                for punto_j in puntos_j:
                    dist = np.linalg.norm(punto_i - punto_j)
                    if dist < separacionMin:
                        separacionMin = dist

    diametroMax = 0.0
    for i in range(numClusters):
        puntos_cluster = datosEscalados[etiquetasPredichas == i]
        if len(puntos_cluster) < 2:
            continue
        for k in range(len(puntos_cluster)):
            for l in range(k + 1, len(puntos_cluster)):
                dist = np.linalg.norm(puntos_cluster[k] - puntos_cluster[l])
                if dist > diametroMax:
                    diametroMax = dist

    dunn = separacionMin / diametroMax if diametroMax > 0 and separacionMin != float('inf') else 0.0

    # Métricas externas
    ari = adjusted_rand_score(etiquetasShuffled, etiquetasPredichas)
    ami = adjusted_mutual_info_score(etiquetasShuffled, etiquetasPredichas)
    nmi = normalized_mutual_info_score(etiquetasShuffled, etiquetasPredichas)

    # Distribución de clusters
    distribucion = {}
    for c in range(numClusters):
        etiquetasCluster = etiquetasShuffled[etiquetasPredichas == c]
        conteo = {e: int(np.sum(etiquetasCluster == e)) for e in np.unique(etiquetasCluster)}
        total_cluster = int(etiquetasCluster.shape[0])
        distribucion[f'cluster_{c}'] = {
            'etiquetasReales': conteo,
            'total': total_cluster
        }

    tamaniosActuales = [int(np.sum(etiquetasPredichas == i)) for i in range(numClusters)]
    primeras5EtiquetasStr = ', '.join(etiquetasShuffled[:5])

    return {
        'silueta': silueta,
        'dunn': dunn,
        'ari': ari,
        'ami': ami,
        'nmi': nmi,
        'distribucion': distribucion,
        'tamaniosActuales': tamaniosActuales,
        'tamaniosMaximos': tamaniosMaximos,
        'primeras5Etiquetas': primeras5EtiquetasStr
    }

def ejecutarExperimentos(datosOriginales, etiquetasOriginales, numSemillas=10, semillaInicial=0):
    """
    Ejecuta múltiples experimentos con diferentes semillas para ambos métodos de clustering.
    Compara CluORT vs STREAMKMeans sobre el dataset Iris.
    
    Args:
        datosOriginales (ndarray): Dataset original [n_samples x n_features]
        etiquetasOriginales (ndarray): Etiquetas verdaderas [n_samples]
        numSemillas (int, optional): Cantidad de semillas a ejecutar. Default: 10
        semillaInicial (int, optional): Semilla inicial para el rango. Default: 0
            Genera semillas desde semillaInicial hasta semillaInicial + numSemillas - 1
    
    Returns:
        tuple: (resultadosCluORT, resultadosKMeans)
            - resultadosCluORT (list): Lista de dicts con resultados de CluORT por semilla
            - resultadosKMeans (list): Lista de dicts con resultados de STREAMKMeans por semilla
            Cada dict contiene: metodo, semilla, silueta, dunn, ari, ami, nmi, distribucion, tamaniosActuales, tamaniosMaximos, primeras5Etiquetas
    """
    print(f"\n{'='*100}")
    print(f"EJECUTANDO {numSemillas} EXPERIMENTOS - COMPARACIÓN CluORT vs STREAMKMeans")
    print(f"Rango de semillas: {semillaInicial} a {semillaInicial + numSemillas - 1}")
    print(f"{'='*100}")
    
    resultadosCluORT = []
    resultadosKMeans = []
    
    for semilla in range(semillaInicial, semillaInicial + numSemillas):
        print(f"\nSemilla {semilla}...")
        
        # Ejecutar CluORT
        print("  CluORT: ", end="")
        resultadoCluORT = ejecutarClusteringConSemilla(datosOriginales, etiquetasOriginales, semilla)
        resultadosCluORT.append({
            'metodo': 'CluORT',
            'semilla': semilla,
            'silueta': resultadoCluORT['silueta'],
            'dunn': resultadoCluORT['dunn'],
            'ari': resultadoCluORT['ari'],
            'ami': resultadoCluORT['ami'],
            'nmi': resultadoCluORT['nmi'],
            'distribucion': resultadoCluORT['distribucion'],
            'tamaniosActuales': resultadoCluORT['tamaniosActuales'],
            'tamaniosMaximos': resultadoCluORT['tamaniosMaximos'],
            'primeras5Etiquetas': resultadoCluORT['primeras5Etiquetas']
        })
        print(f"Silueta: {resultadoCluORT['silueta']:.4f} | ARI: {resultadoCluORT['ari']:.4f}")
        
        # Ejecutar STREAMKMeans
        print("  STREAMKMeans: ", end="")
        resultadoKMeans = ejecutarSTREAMKMeansConSemilla(datosOriginales, etiquetasOriginales, semilla)
        resultadosKMeans.append({
            'metodo': 'STREAMKMeans',
            'semilla': semilla,
            'silueta': resultadoKMeans['silueta'],
            'dunn': resultadoKMeans['dunn'],
            'ari': resultadoKMeans['ari'],
            'ami': resultadoKMeans['ami'],
            'nmi': resultadoKMeans['nmi'],
            'distribucion': resultadoKMeans['distribucion'],
            'tamaniosActuales': resultadoKMeans['tamaniosActuales'],
            'tamaniosMaximos': resultadoKMeans['tamaniosMaximos'],
            'primeras5Etiquetas': resultadoKMeans['primeras5Etiquetas']
        })
        print(f"Silueta: {resultadoKMeans['silueta']:.4f} | ARI: {resultadoKMeans['ari']:.4f}")
    
    return resultadosCluORT, resultadosKMeans


def crearTablaResumen(resultadosCluORT, resultadosKMeans, dirResultados):
    """
    Crea y guarda tabla resumen de resultados para ambos métodos de clustering.
    Genera archivos PNG (tablas formateadas) y CSV (datos brutos).
    
    Args:
        resultadosCluORT (list): Resultados del método CluORT
        resultadosKMeans (list): Resultados del método STREAMKMeans
        dirResultados (Path): Directorio donde guardar los archivos de salida
    
    Returns:
        pd.DataFrame: DataFrame combinado con todos los resultados y métricas
    """
    # Combinar resultados de ambos métodos
    todosResultados = resultadosCluORT + resultadosKMeans
    dfResultados = pd.DataFrame(todosResultados)
    
    # Mapeo de nombres completos a abreviados
    mapeoAbreviado = {'setosa': 'set', 'versicolor': 'ver', 'virginica': 'vir'}
    
    # Añadir columna de distribución por cluster
    def formatearDistribucion(distribucion):
        clusters = []
        for clusterKey in sorted(distribucion.keys()):
            clusterIdx = int(clusterKey.split('_')[1])
            info = distribucion[clusterKey]
            etiquetas = info.get("etiquetas_reales", info.get("etiquetasReales"))
            # Calcular total de elementos en el cluster (ignorando 'sinEtiqueta')
            total_cluster = distribucion[clusterKey].get('total') if isinstance(distribucion[clusterKey], dict) else None
            if total_cluster is None:
                total_cluster = sum(count for label, count in etiquetas.items() if label != 'sinEtiqueta')
            
            # Filtrar etiquetas None y formatear
            etiquetasStr = ' '.join([f"{mapeoAbreviado.get(label, label)}:{count}" 
                                      for label, count in sorted(etiquetas.items()) 
                                      if label != 'sinEtiqueta'])
            
            clusters.append(f"C{clusterIdx} (n={total_cluster}): {etiquetasStr}")
        
        return '<br>'.join(clusters)
    
    dfResultados['capacidad'] = dfResultados['distribucion'].apply(formatearDistribucion)
    
    # Redondear métricas a 3 decimales
    for col in ['silueta', 'dunn', 'ari', 'ami', 'nmi']:
        dfResultados[col] = dfResultados[col].round(3)
    
    # Guardar como CSV
    archivoCSV = dirResultados / "resumen_experimentos.csv"
    dfResultados.to_csv(archivoCSV, index=False)
    
    # Crear tabla formateada para mostrar
    print(f"\n{'='*100}")
    print(f"RESUMEN DE RESULTADOS - TODAS LAS MÉTRICAS")
    print(f"{'='*100}")
    print("\nMÉTRICAS INTERNAS:")
    print(dfResultados[['metodo', 'semilla', 'silueta', 'dunn', 'capacidad']].to_string(index=False))
    print("\nMÉTRICAS EXTERNAS:")
    print(dfResultados[['metodo', 'semilla', 'ari', 'ami', 'nmi', 'capacidad']].to_string(index=False))
    print(f"\n\u2713 Tabla guardada en: {archivoCSV}")
    
    # Guardar tabla como imagen con dataframe_image (formato booktabs LaTeX)
    # Crear tabla separada para cada método
    dfCluORT = dfResultados[dfResultados['metodo'] == 'CluORT'][['semilla', 'silueta', 'dunn', 'ari', 'ami', 'nmi', 'capacidad', 'primeras5Etiquetas']].copy()
    dfCluORT.columns = ['Semilla', 'Silueta', 'Dunn', 'ARI', 'AMI', 'NMI', 'Capacidad', 'Etiquetas Iniciales']
    dfCluORT.insert(0, 'Método', 'CluORT')
    
    dfKMeans = dfResultados[dfResultados['metodo'] == 'STREAMKMeans'][['semilla', 'silueta', 'dunn', 'ari', 'ami', 'nmi', 'capacidad', 'primeras5Etiquetas']].copy()
    dfKMeans.columns = ['Semilla', 'Silueta', 'Dunn', 'ARI', 'AMI', 'NMI', 'Capacidad', 'Etiquetas Iniciales']
    dfKMeans.insert(0, 'Método', 'STREAMKMeans')
    
    # Combinar ambas tablas
    dfTablaCompleta = pd.concat([dfCluORT, dfKMeans], ignore_index=True)
    
    # Reemplazar <br> por saltos de línea para dataframe_image
    dfTablaCompleta['Capacidad'] = dfTablaCompleta['Capacidad'].str.replace('<br>', '\n', regex=False)
    
    # Aplicar estilo booktabs: solo líneas horizontales, sin verticales
    styled = dfTablaCompleta.style.set_properties(**{
        'text-align': 'center',
        'font-size': '9pt',
        'font-family': 'Arial',
        'white-space': 'pre-wrap'
    }).set_table_styles([
        # Línea superior gruesa
        {'selector': 'thead th', 'props': [
            ('border-top', '2px solid black'),
            ('border-bottom', '2px solid black'),
            ('border-left', 'none'),
            ('border-right', 'none'),
            ('background-color', 'white'),
            ('font-weight', 'bold')
        ]},
        # Líneas finas entre filas
        {'selector': 'tbody td', 'props': [
            ('border-top', '0.5px solid #cccccc'),
            ('border-left', 'none'),
            ('border-right', 'none')
        ]},
        # Línea inferior gruesa
        {'selector': 'tbody tr:last-child td', 'props': [
            ('border-bottom', '2px solid black')
        ]},
        # Sin bordes laterales en la tabla
        {'selector': '', 'props': [
            ('border-left', 'none'),
            ('border-right', 'none')
        ]}
    ]).hide(axis='index')
    
    # Guardar tabla conjunta
    archivoTablaImg = dirResultados / "tabla_resumen.png"
    dfi.export(styled, str(archivoTablaImg), dpi=300)
    print(f"✓ Tabla conjunta guardada en: {archivoTablaImg}")
    
    # Guardar tabla separada de CluORT
    dfCluORTTabla = dfCluORT[['Semilla', 'Silueta', 'Dunn', 'ARI', 'AMI', 'NMI', 'Capacidad', 'Etiquetas Iniciales']].copy()
    dfCluORTTabla.insert(0, 'Método', 'CluORT')
    styledCluORT = dfCluORTTabla.style.set_properties(**{
        'text-align': 'center',
        'font-size': '9pt',
        'font-family': 'Arial',
        'white-space': 'pre-wrap'
    }).set_table_styles([
        {'selector': 'thead th', 'props': [
            ('border-top', '2px solid black'),
            ('border-bottom', '2px solid black'),
            ('border-left', 'none'),
            ('border-right', 'none'),
            ('background-color', 'white'),
            ('font-weight', 'bold')
        ]},
        {'selector': 'tbody td', 'props': [
            ('border-top', '0.5px solid #cccccc'),
            ('border-left', 'none'),
            ('border-right', 'none')
        ]},
        {'selector': 'tbody tr:last-child td', 'props': [
            ('border-bottom', '2px solid black')
        ]},
        {'selector': '', 'props': [
            ('border-left', 'none'),
            ('border-right', 'none')
        ]}
    ]).hide(axis='index')
    
    archivoTablaCluORT = dirResultados / "tabla_resumen_cluort.png"
    dfi.export(styledCluORT, str(archivoTablaCluORT), dpi=300)
    print(f"✓ Tabla CluORT separada guardada en: {archivoTablaCluORT}")
    
    # Guardar tabla separada de KMeans
    dfKMeansTabla = dfKMeans[['Semilla', 'Silueta', 'Dunn', 'ARI', 'AMI', 'NMI', 'Capacidad', 'Etiquetas Iniciales']].copy()
    dfKMeansTabla.insert(0, 'Método', 'STREAMKMeans')
    styledKMeans = dfKMeansTabla.style.set_properties(**{
        'text-align': 'center',
        'font-size': '9pt',
        'font-family': 'Arial',
        'white-space': 'pre-wrap'
    }).set_table_styles([
        {'selector': 'thead th', 'props': [
            ('border-top', '2px solid black'),
            ('border-bottom', '2px solid black'),
            ('border-left', 'none'),
            ('border-right', 'none'),
            ('background-color', 'white'),
            ('font-weight', 'bold')
        ]},
        {'selector': 'tbody td', 'props': [
            ('border-top', '0.5px solid #cccccc'),
            ('border-left', 'none'),
            ('border-right', 'none')
        ]},
        {'selector': 'tbody tr:last-child td', 'props': [
            ('border-bottom', '2px solid black')
        ]},
        {'selector': '', 'props': [
            ('border-left', 'none'),
            ('border-right', 'none')
        ]}
    ]).hide(axis='index')
    
    archivoTablaKMeans = dirResultados / "tabla_resumen_kmeans.png"
    dfi.export(styledKMeans, str(archivoTablaKMeans), dpi=300)
    print(f"✓ Tabla STREAMKMeans separada guardada en: {archivoTablaKMeans}")
    
    return dfResultados


def graficarEvolucionMetricas(dfResultados, dirResultados):
    """
    Genera gráfico de evolución de métricas por semilla comparando ambos métodos.
    Muestra 5 subplots (uno por métrica) con líneas para CluORT y STREAMKMeans.
    
    Args:
        dfResultados (pd.DataFrame): DataFrame con columnas: metodo, semilla, silueta, dunn, ari, ami, nmi
        dirResultados (Path): Directorio donde guardar el gráfico PNG
    
    Returns:
        None (guarda archivo PNG)
    """
    metricas = ['silueta', 'dunn', 'ari', 'ami', 'nmi']
    numMetricas = len(metricas)
    nCols = 3
    nFilas = (numMetricas + nCols - 1) // nCols
    
    fig, axes = plt.subplots(nFilas, nCols, figsize=(15, nFilas * 5))
    fig.suptitle('Comparación CluORT vs STREAMKMeans - Evolución de Métricas', fontsize=16, fontweight='bold')
    
    # Generar colores para los métodos
    colorCluORT = '#E15759'  # Rojo
    colorSTREAMKMeans = '#4E79A7'  # Azul
    
    axes = axes.flatten() if nFilas > 1 else [axes]
    
    # Separar datos por método
    dfCluORT = dfResultados[dfResultados['metodo'] == 'CluORT']
    dfSTREAMKMeans = dfResultados[dfResultados['metodo'] == 'STREAMKMeans']
    
    for idx, metrica in enumerate(metricas):
        ax = axes[idx]
        
        # Plot CluORT
        ax.plot(dfCluORT['semilla'], dfCluORT[metrica], 
                marker='o', linewidth=2, markersize=8, color=colorCluORT, label='CluORT', alpha=0.8)
        
        # Plot STREAMKMeans
        ax.plot(dfSTREAMKMeans['semilla'], dfSTREAMKMeans[metrica], 
                marker='s', linewidth=2, markersize=8, color=colorSTREAMKMeans, label='STREAMKMeans', alpha=0.8)
        
        ax.set_xlabel('Semilla Random', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'{metrica.upper()}', fontsize=10, fontweight='bold')
        ax.set_title(f'{metrica.upper()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(dfCluORT['semilla'])
        ax.legend(loc='best')
    
    # Ocultar subplots vacíos
    for idx in range(numMetricas, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    archivoGrafico = dirResultados / "evolucion_metricas.png"
    plt.savefig(archivoGrafico, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de evolución conjunta guardado en: {archivoGrafico}")
    plt.close()


def graficarEvolucionMetricasCluORT(dfResultados, dirResultados):
    """
    Genera gráfico de evolución de métricas solo para el método CluORT.
    Muestra 5 subplots (uno por métrica) con evolución por semilla.
    
    Args:
        dfResultados (pd.DataFrame): DataFrame con resultados de clustering
        dirResultados (Path): Directorio donde guardar el gráfico PNG
    
    Returns:
        None (guarda archivo PNG)
    """
    metricas = ['silueta', 'dunn', 'ari', 'ami', 'nmi']
    numMetricas = len(metricas)
    nCols = 3
    nFilas = (numMetricas + nCols - 1) // nCols
    
    fig, axes = plt.subplots(nFilas, nCols, figsize=(15, nFilas * 5))
    fig.suptitle('CluORT - Evolución de Métricas', fontsize=16, fontweight='bold')
    
    colorCluORT = '#E15759'  # Rojo
    
    axes = axes.flatten() if nFilas > 1 else [axes]
    
    # Filtrar solo CluORT
    dfCluORT = dfResultados[dfResultados['metodo'] == 'CluORT']
    
    for idx, metrica in enumerate(metricas):
        ax = axes[idx]
        
        # Plot CluORT
        ax.plot(dfCluORT['semilla'], dfCluORT[metrica], 
                marker='o', linewidth=2.5, markersize=10, color=colorCluORT, label='CluORT', alpha=0.8)
        
        ax.set_xlabel('Semilla Random', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'{metrica.upper()}', fontsize=10, fontweight='bold')
        ax.set_title(f'{metrica.upper()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(dfCluORT['semilla'])
        ax.legend(loc='best', fontsize=10)
    
    # Ocultar subplots vacíos
    for idx in range(numMetricas, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    archivoGrafico = dirResultados / "evolucion_metricas_cluort.png"
    plt.savefig(archivoGrafico, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de evolución CluORT guardado en: {archivoGrafico}")
    plt.close()


def graficarEvolucionMetricasSTREAMKMeans(dfResultados, dirResultados):
    """
    Genera gráfico de evolución de métricas solo para el método STREAMKMeans.
    Muestra 5 subplots (uno por métrica) con evolución por semilla.
    
    Args:
        dfResultados (pd.DataFrame): DataFrame con resultados de clustering
        dirResultados (Path): Directorio donde guardar el gráfico PNG
    
    Returns:
        None (guarda archivo PNG)
    """
    metricas = ['silueta', 'dunn', 'ari', 'ami', 'nmi']
    numMetricas = len(metricas)
    nCols = 3
    nFilas = (numMetricas + nCols - 1) // nCols
    
    fig, axes = plt.subplots(nFilas, nCols, figsize=(15, nFilas * 5))
    fig.suptitle('STREAMKMeans - Evolución de Métricas', fontsize=16, fontweight='bold')
    
    colorSTREAMKMeans = '#4E79A7'  # Azul
    
    axes = axes.flatten() if nFilas > 1 else [axes]
    
    # Filtrar solo STREAMKMeans
    dfSTREAMKMeans = dfResultados[dfResultados['metodo'] == 'STREAMKMeans']
    
    for idx, metrica in enumerate(metricas):
        ax = axes[idx]
        
        # Plot STREAMKMeans
        ax.plot(dfSTREAMKMeans['semilla'], dfSTREAMKMeans[metrica], 
                marker='s', linewidth=2.5, markersize=10, color=colorSTREAMKMeans, label='STREAMKMeans', alpha=0.8)
        
        ax.set_xlabel('Semilla Random', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'{metrica.upper()}', fontsize=10, fontweight='bold')
        ax.set_title(f'{metrica.upper()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(dfSTREAMKMeans['semilla'])
        ax.legend(loc='best', fontsize=10)
    
    # Ocultar subplots vacíos
    for idx in range(numMetricas, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    archivoGrafico = dirResultados / "evolucion_metricas_streamkmeans.png"
    plt.savefig(archivoGrafico, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de evolución STREAMKMeans guardado en: {archivoGrafico}")
    plt.close()


def graficarDistribucionConjunta(resultadosCluORT, resultadosKMeans, datosOriginales, etiquetasOriginales, dirResultados):
    """
    Genera gráfico de distribución conjunta: CluORT vs STREAMKMeans lado a lado por semilla.
    Muestra histogramas de distribución de clases reales dentro de cada cluster predicho.
    
    Args:
        resultadosCluORT (list): Resultados del método CluORT
        resultadosKMeans (list): Resultados del método STREAMKMeans
        datosOriginales (ndarray): Dataset original
        etiquetasOriginales (ndarray): Etiquetas reales de clase
        dirResultados (Path): Directorio donde guardar el gráfico PNG
    
    Returns:
        None (guarda archivo PNG)
    """
    numSemillas = len(resultadosCluORT)
    numClusters = len(resultadosCluORT[0]['tamaniosActuales'])
    
    # Obtener clases únicas
    clasesUnicas = sorted(np.unique(etiquetasOriginales))
    numClases = len(clasesUnicas)
    
    # Crear figura: numSemillas filas x 2 columnas (CluORT, STREAMKMeans)
    fig, axes = plt.subplots(numSemillas, 2, figsize=(16, 2.5 * numSemillas))
    if numSemillas == 1:
        axes = axes.reshape(1, -1)
    
    # Generar paleta de colores
    if numClases <= 10:
        cmap = plt.colormaps.get_cmap('tab10')
    elif numClases <= 20:
        cmap = plt.colormaps.get_cmap('tab20')
    else:
        cmap = plt.colormaps.get_cmap('hsv')
    coloresClases = [cmap(i) for i in np.linspace(0, 0.9, numClases)]
    
    # Procesar cada semilla
    for idxSemilla in range(numSemillas):
        resultado_cluort = resultadosCluORT[idxSemilla]
        resultado_kmeans = resultadosKMeans[idxSemilla]
        
        # CluORT en columna izquierda
        ax_cluort = axes[idxSemilla, 0]
        
        # STREAMKMeans en columna derecha
        ax_kmeans = axes[idxSemilla, 1]
        
        # ===== PROCESAR CluORT =====
        matrizDistribucion_cluort = np.zeros((numClusters, numClases))
        distribucionDict = resultado_cluort['distribucion']
        
        for clusterKey, clusterData in distribucionDict.items():
            idxCluster = int(clusterKey.split('_')[1])
            etiquetasReales = clusterData['etiquetasReales']
            
            for clase, cantidad in etiquetasReales.items():
                if clase != 'sinEtiqueta' and clase in clasesUnicas:
                    idxClase = clasesUnicas.index(clase)
                    matrizDistribucion_cluort[idxCluster, idxClase] = cantidad
        
        # Graficar CluORT
        posicionesY = np.arange(numClusters)
        acumulado = np.zeros(numClusters)
        
        for idxClase, clase in enumerate(clasesUnicas):
            valores = matrizDistribucion_cluort[:, idxClase]
            
            barras = ax_cluort.barh(posicionesY, valores, left=acumulado,
                           color=coloresClases[idxClase],
                           alpha=0.85, edgecolor='black', linewidth=0.6)
            
            for i, (barra, valor) in enumerate(zip(barras, valores)):
                if valor > 0:
                    xPos = acumulado[i] + valor / 2
                    ax_cluort.text(xPos, barra.get_y() + barra.get_height()/2.,
                           f'{int(valor)}',
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           color='black', 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                   edgecolor='none', alpha=0.7))
            
            acumulado += valores
        
        # Configurar CluORT
        # Etiquetas de eje Y con total por cluster
        totales_cluort = matrizDistribucion_cluort.sum(axis=1)
        etiquetas_cluort = [f"{i} (n={int(totales_cluort[i])})" for i in range(numClusters)]
        ax_cluort.set_yticks(posicionesY)
        ax_cluort.set_yticklabels(etiquetas_cluort, fontsize=11)
        ax_cluort.tick_params(axis='x', labelsize=9)
        ax_cluort.tick_params(axis='y', labelsize=11)
        ax_cluort.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.3)
        maxValor = matrizDistribucion_cluort.sum(axis=1).max()
        ax_cluort.set_xlim(0, maxValor)
        ax_cluort.invert_yaxis()
        
        # Título CluORT con métricas
        metricasTexto = f"Semilla {resultado_cluort['semilla']}  |  Sil: {resultado_cluort['silueta']:.2f}  |  D: {resultado_cluort['dunn']:.2f}  |  ARI: {resultado_cluort['ari']:.2f}  |  AMI: {resultado_cluort['ami']:.2f}  |  NMI: {resultado_cluort['nmi']:.2f}"
        ax_cluort.text(0.5, 1.12, metricasTexto, 
                      transform=ax_cluort.transAxes, fontsize=10, fontweight='bold',
                      va='bottom', ha='center',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5E5', edgecolor='#E15759', linewidth=1.5, alpha=0.95))
        
        # ===== PROCESAR STREAMKMEANS =====
        matrizDistribucion_kmeans = np.zeros((numClusters, numClases))
        distribucionDict = resultado_kmeans['distribucion']
        
        for clusterKey, clusterData in distribucionDict.items():
            idxCluster = int(clusterKey.split('_')[1])
            etiquetasReales = clusterData['etiquetasReales']
            
            for clase, cantidad in etiquetasReales.items():
                if clase != 'sinEtiqueta' and clase in clasesUnicas:
                    idxClase = clasesUnicas.index(clase)
                    matrizDistribucion_kmeans[idxCluster, idxClase] = cantidad
        
        # Graficar STREAMKMeans
        acumulado = np.zeros(numClusters)
        
        for idxClase, clase in enumerate(clasesUnicas):
            valores = matrizDistribucion_kmeans[:, idxClase]
            
            barras = ax_kmeans.barh(posicionesY, valores, left=acumulado,
                           color=coloresClases[idxClase],
                           alpha=0.85, edgecolor='black', linewidth=0.6)
            
            for i, (barra, valor) in enumerate(zip(barras, valores)):
                if valor > 0:
                    xPos = acumulado[i] + valor / 2
                    ax_kmeans.text(xPos, barra.get_y() + barra.get_height()/2.,
                           f'{int(valor)}',
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           color='black', 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                   edgecolor='none', alpha=0.7))
            
            acumulado += valores
        
        # Configurar STREAMKMeans
        # Etiquetas de eje Y con total por cluster
        totales_kmeans = matrizDistribucion_kmeans.sum(axis=1)
        etiquetas_kmeans = [f"{i} (n={int(totales_kmeans[i])})" for i in range(numClusters)]
        ax_kmeans.set_yticks(posicionesY)
        ax_kmeans.set_yticklabels(etiquetas_kmeans, fontsize=11)
        ax_kmeans.tick_params(axis='x', labelsize=9)
        ax_kmeans.tick_params(axis='y', labelsize=11)
        ax_kmeans.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.3)
        maxValor = matrizDistribucion_kmeans.sum(axis=1).max()
        ax_kmeans.set_xlim(0, maxValor)
        ax_kmeans.invert_yaxis()
        
        # Título STREAMKMeans con métricas
        metricasTexto = f"Semilla {resultado_kmeans['semilla']}  |  Sil: {resultado_kmeans['silueta']:.2f}  |  D: {resultado_kmeans['dunn']:.2f}  |  ARI: {resultado_kmeans['ari']:.2f}  |  AMI: {resultado_kmeans['ami']:.2f}  |  NMI: {resultado_kmeans['nmi']:.2f}"
        ax_kmeans.text(0.5, 1.12, metricasTexto,
                      transform=ax_kmeans.transAxes, fontsize=10, fontweight='bold',
                      va='bottom', ha='center',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='#E5F5FF', edgecolor='#4E79A7', linewidth=1.5, alpha=0.95))
    
    # Crear leyenda de clases
    leyendaElements = [plt.Rectangle((0, 0), 1, 1, fc=coloresClases[i], alpha=0.85, edgecolor='black', linewidth=1) 
                       for i in range(numClases)]
    fig.legend(leyendaElements, [c.capitalize() for c in clasesUnicas], 
              loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=numClases, 
              fontsize=11, frameon=True, fancybox=True, shadow=False)
    
    plt.suptitle('Distribución de Clases por Cluster - CluORT vs STREAMKMeans', 
                fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.subplots_adjust(hspace=0.65, wspace=0.22, top=0.88)
    
    archivoGrafico = dirResultados / "distribucion_clases_conjunta.png"
    plt.savefig(archivoGrafico, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de distribución conjunta guardado en: {archivoGrafico}")
    plt.close()


def graficarRestriccionesTamanio(resultados, datosOriginales, etiquetasOriginales, dirResultados, sufijo=''):
    """
    Genera gráfico mostrando la distribución de clases reales por cluster para cada semilla.
    Visualiza restricciones de tamaño y composición de clases en cada cluster.
    
    Args:
        resultados (list): Resultados de un método de clustering (CluORT o STREAMKMeans)
        datosOriginales (ndarray): Dataset original
        etiquetasOriginales (ndarray): Etiquetas reales de clase
        dirResultados (Path): Directorio donde guardar el gráfico PNG
        sufijo (str, optional): Sufijo para el nombre del archivo. Default: ''
    
    Returns:
        None (guarda archivo PNG)
    """
    metodo = resultados[0].get('metodo', 'Unknown')
    numSemillas = len(resultados)
    numClusters = len(resultados[0]['tamaniosActuales'])
    
    # Obtener clases únicas
    clasesUnicas = sorted(np.unique(etiquetasOriginales))
    numClases = len(clasesUnicas)
    
    # Crear figura con subplots por semilla (2 columnas)
    nFilas = (numSemillas + 1) // 2
    nCols = 2
    fig, axes = plt.subplots(nFilas, nCols, figsize=(16, nFilas * 4))
    axes = axes.flatten() if numSemillas > 1 else [axes]
    
    # Generar paleta de colores dinámica según el número de clases
    if numClases <= 10:
        cmap = plt.colormaps.get_cmap('tab10')
    elif numClases <= 20:
        cmap = plt.colormaps.get_cmap('tab20')
    else:
        cmap = plt.colormaps.get_cmap('hsv')
    coloresClases = [cmap(i) for i in np.linspace(0, 0.9, numClases)]
    
    # Procesar cada semilla
    for idxSemilla, resultado in enumerate(resultados):
        ax = axes[idxSemilla]
        
        # Crear matriz de distribución: clusters x clases
        matrizDistribucion = np.zeros((numClusters, numClases))
        
        # Obtener distribución del diccionario
        distribucionDict = resultado['distribucion']
        
        for clusterKey, clusterData in distribucionDict.items():
            idxCluster = int(clusterKey.split('_')[1])
            etiquetasReales = clusterData['etiquetasReales']
            
            # Para cada clase real en el cluster
            for clase, cantidad in etiquetasReales.items():
                if clase != 'sinEtiqueta' and clase in clasesUnicas:
                    idxClase = clasesUnicas.index(clase)
                    matrizDistribucion[idxCluster, idxClase] = cantidad
        
        # Crear barras apiladas horizontales
        posicionesY = np.arange(numClusters)
        acumulado = np.zeros(numClusters)
        
        for idxClase, clase in enumerate(clasesUnicas):
            valores = matrizDistribucion[:, idxClase]
            
            barras = ax.barh(posicionesY, valores, left=acumulado,
                           label=clase.capitalize(), color=coloresClases[idxClase],
                           alpha=0.85, edgecolor='black', linewidth=1.2)
            
            # Agregar valores en el centro de cada segmento de barra
            for i, (barra, valor) in enumerate(zip(barras, valores)):
                if valor > 0:
                    # Posición X en el centro del segmento
                    xPos = acumulado[i] + valor / 2
                    ax.text(xPos, barra.get_y() + barra.get_height()/2.,
                           f'{int(valor)}',
                           ha='center', va='center', fontsize=10, fontweight='bold',
                           color='black', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor='none', alpha=0.7))
            
            # Actualizar acumulado para la siguiente clase
            acumulado += valores
        
        # Configurar ejes
        ax.set_ylabel('Cluster', fontsize=11, fontweight='bold')
        ax.set_xlabel('Cantidad de Puntos', fontsize=11, fontweight='bold')
        
        # Título con métricas
        metricasTexto = f"Semilla {resultado['semilla']}  |  Sil: {resultado['silueta']:.2f}  |  D: {resultado['dunn']:.2f}  |  ARI: {resultado['ari']:.2f}  |  AMI: {resultado['ami']:.2f}  |  NMI: {resultado['nmi']:.2f}"
        colorFondo = '#FFE5E5' if metodo == 'CluORT' else '#E5F5FF'
        colorBorde = '#E15759' if metodo == 'CluORT' else '#4E79A7'
        
        ax.text(0.5, 1.15, metricasTexto, 
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                va='bottom', ha='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=colorFondo, edgecolor=colorBorde, linewidth=1.5, alpha=0.95))
        
        ax.set_yticks(posicionesY)
        # Etiquetas de eje Y con total de puntos por cluster
        totales = matrizDistribucion.sum(axis=1)
        etiquetas_clusters = [f"C{i} (n={int(totales[i])})" for i in range(numClusters)]
        ax.set_yticklabels(etiquetas_clusters)
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # Calcular máximo dinámico desde los datos reales
        maxValor = matrizDistribucion.sum(axis=1).max()
        ax.set_xlim(0, maxValor * 1.1)
        ax.invert_yaxis()  # C0 arriba, C2 abajo
    
    # Ocultar subplots vacíos si hay un número impar de semillas
    for idx in range(numSemillas, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'{metodo} - Distribución de Clases por Cluster y Semilla', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    archivoGrafico = dirResultados / f"distribucion_clases_por_cluster{sufijo}.png"
    plt.savefig(archivoGrafico, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de distribución de clases guardado en: {archivoGrafico}")
    plt.close()


def graficarComparativaAmbosMetodos(resultadosCluORT, resultadosKMeans, datosOriginales, etiquetasOriginales, dirResultados):
    """
    Genera gráfico comparativo: CluORT vs STREAMKMeans lado a lado por semilla.
    Organizado como filas (semillas) x 2 columnas (CluORT | STREAMKMeans).
    
    Args:
        resultadosCluORT (list): Resultados del método CluORT por semilla
        resultadosKMeans (list): Resultados del método STREAMKMeans por semilla
        datosOriginales (ndarray): Dataset original
        etiquetasOriginales (ndarray): Etiquetas reales de clase
        dirResultados (Path): Directorio donde guardar el gráfico PNG
    
    Returns:
        None (guarda archivo PNG)
    """
    numSemillas = len(resultadosCluORT)
    numClusters = len(resultadosCluORT[0]['tamaniosActuales'])
    
    # Obtener clases únicas
    clasesUnicas = sorted(np.unique(etiquetasOriginales))
    numClases = len(clasesUnicas)
    
    # Crear figura: numSemillas filas x 2 columnas (CluORT, STREAMKMeans) - MÁS COMPACTA
    fig, axes = plt.subplots(numSemillas, 2, figsize=(16, 2.5 * numSemillas))
    if numSemillas == 1:
        axes = axes.reshape(1, -1)
    
    # Generar paleta de colores
    if numClases <= 10:
        cmap = plt.colormaps.get_cmap('tab10')
    elif numClases <= 20:
        cmap = plt.colormaps.get_cmap('tab20')
    else:
        cmap = plt.colormaps.get_cmap('hsv')
    coloresClases = [cmap(i) for i in np.linspace(0, 0.9, numClases)]
    
    # Procesar cada semilla
    for idxSemilla in range(numSemillas):
        resultado_cluort = resultadosCluORT[idxSemilla]
        resultado_kmeans = resultadosKMeans[idxSemilla]
        
        # CluORT en columna izquierda
        ax_cluort = axes[idxSemilla, 0]
        
        # STREAMKMeans en columna derecha
        ax_kmeans = axes[idxSemilla, 1]
        
        # ===== PROCESAR CluORT =====
        matrizDistribucion_cluort = np.zeros((numClusters, numClases))
        distribucionDict = resultado_cluort['distribucion']
        
        for clusterKey, clusterData in distribucionDict.items():
            idxCluster = int(clusterKey.split('_')[1])
            etiquetasReales = clusterData['etiquetasReales']
            
            for clase, cantidad in etiquetasReales.items():
                if clase != 'sinEtiqueta' and clase in clasesUnicas:
                    idxClase = clasesUnicas.index(clase)
                    matrizDistribucion_cluort[idxCluster, idxClase] = cantidad
        
        # Graficar CluORT
        posicionesY = np.arange(numClusters)
        acumulado = np.zeros(numClusters)
        
        for idxClase, clase in enumerate(clasesUnicas):
            valores = matrizDistribucion_cluort[:, idxClase]
            
            barras = ax_cluort.barh(posicionesY, valores, left=acumulado,
                           color=coloresClases[idxClase],
                           alpha=0.85, edgecolor='black', linewidth=0.6)
            
            for i, (barra, valor) in enumerate(zip(barras, valores)):
                if valor > 0:
                    xPos = acumulado[i] + valor / 2
                    ax_cluort.text(xPos, barra.get_y() + barra.get_height()/2.,
                           f'{int(valor)}',
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           color='black', 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                   edgecolor='none', alpha=0.7))
            
            acumulado += valores
        
        # Configurar CluORT
        # Etiquetas de eje Y con total por cluster
        totales_cluort = matrizDistribucion_cluort.sum(axis=1)
        etiquetas_cluort = [f"{i} (n={int(totales_cluort[i])})" for i in range(numClusters)]
        ax_cluort.set_yticks(posicionesY)
        ax_cluort.set_yticklabels(etiquetas_cluort, fontsize=11)
        ax_cluort.tick_params(axis='x', labelsize=9)
        ax_cluort.tick_params(axis='y', labelsize=11)
        ax_cluort.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.3)
        maxValor = matrizDistribucion_cluort.sum(axis=1).max()
        ax_cluort.set_xlim(0, maxValor)
        ax_cluort.invert_yaxis()
        
        # Agregar métricas CluORT como título centrado en la parte superior
        silueta = resultado_cluort['silueta']
        dunn = resultado_cluort['dunn']
        ari = resultado_cluort['ari']
        ami = resultado_cluort['ami']
        nmi = resultado_cluort['nmi']
        
        metricasTexto = f"Semilla {resultado_cluort['semilla']}  |  Sil: {silueta:.2f}  |  D: {dunn:.2f}  |  ARI: {ari:.2f}  |  AMI: {ami:.2f}  |  NMI: {nmi:.2f}"
        ax_cluort.text(0.5, 1.12, metricasTexto, 
                      transform=ax_cluort.transAxes, fontsize=11, fontweight='bold',
                      va='bottom', ha='center',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5E5', edgecolor='#E15759', linewidth=1.5, alpha=0.95))
        
        # ===== PROCESAR STREAMKMEANS =====
        matrizDistribucion_kmeans = np.zeros((numClusters, numClases))
        distribucionDict = resultado_kmeans['distribucion']
        
        for clusterKey, clusterData in distribucionDict.items():
            idxCluster = int(clusterKey.split('_')[1])
            etiquetasReales = clusterData['etiquetasReales']
            
            for clase, cantidad in etiquetasReales.items():
                if clase != 'sinEtiqueta' and clase in clasesUnicas:
                    idxClase = clasesUnicas.index(clase)
                    matrizDistribucion_kmeans[idxCluster, idxClase] = cantidad
        
        # Graficar STREAMKMeans
        acumulado = np.zeros(numClusters)
        
        for idxClase, clase in enumerate(clasesUnicas):
            valores = matrizDistribucion_kmeans[:, idxClase]
            
            barras = ax_kmeans.barh(posicionesY, valores, left=acumulado,
                           color=coloresClases[idxClase],
                           alpha=0.85, edgecolor='black', linewidth=0.6)
            
            for i, (barra, valor) in enumerate(zip(barras, valores)):
                if valor > 0:
                    xPos = acumulado[i] + valor / 2
                    ax_kmeans.text(xPos, barra.get_y() + barra.get_height()/2.,
                           f'{int(valor)}',
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           color='black', 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                   edgecolor='none', alpha=0.7))
            
            acumulado += valores
        
        # Configurar STREAMKMeans
        # Etiquetas de eje Y con total por cluster
        totales_kmeans = matrizDistribucion_kmeans.sum(axis=1)
        etiquetas_kmeans = [f"{i} (n={int(totales_kmeans[i])})" for i in range(numClusters)]
        ax_kmeans.set_yticks(posicionesY)
        ax_kmeans.set_yticklabels(etiquetas_kmeans, fontsize=11)
        ax_kmeans.tick_params(axis='x', labelsize=9)
        ax_kmeans.tick_params(axis='y', labelsize=11)
        ax_kmeans.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.3)
        maxValor = matrizDistribucion_kmeans.sum(axis=1).max()
        ax_kmeans.set_xlim(0, maxValor)
        ax_kmeans.invert_yaxis()
        
        # Agregar métricas STREAMKMeans
        silueta = resultado_kmeans['silueta']
        dunn = resultado_kmeans['dunn']
        ari = resultado_kmeans['ari']
        ami = resultado_kmeans['ami']
        nmi = resultado_kmeans['nmi']
        
        # Poner semilla y todas las métricas como título centrado en la parte superior
        metricasTexto = f"Semilla {resultado_kmeans['semilla']}  |  Sil: {silueta:.2f}  |  D: {dunn:.2f}  |  ARI: {ari:.2f}  |  AMI: {ami:.2f}  |  NMI: {nmi:.2f}"
        ax_kmeans.text(0.5, 1.12, metricasTexto,
               transform=ax_kmeans.transAxes, fontsize=11, fontweight='bold',
               va='bottom', ha='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#E5F5FF', edgecolor='#4E79A7', linewidth=1.5, alpha=0.95))
    
    # Crear leyenda de clases como elementos separados
    leyendaElements = [plt.Rectangle((0, 0), 1, 1, fc=coloresClases[i], alpha=0.85, edgecolor='black', linewidth=1) 
                       for i in range(numClases)]
    fig.legend(leyendaElements, [c.capitalize() for c in clasesUnicas], 
              loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=numClases, 
              fontsize=11, frameon=True, fancybox=True, shadow=False)
    
    plt.suptitle('CluORT vs STREAMKMeans', 
                fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.subplots_adjust(hspace=0.55, wspace=0.22, top=0.88)
    
    archivoGrafico = dirResultados / "comparativa_cluort_vs_streamkmeans.png"
    plt.savefig(archivoGrafico, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico comparativo CluORT vs STREAMKMeans guardado en: {archivoGrafico}")
    plt.close()


def main():
    """
    Función principal que orquesta todo el pipeline de experimentación.
    Ejecuta clustering con CluORT y STREAMKMeans, genera visualizaciones y reportes.
    
    Args:
        None
    
    Returns:
        None (genera archivos PNG y CSV en directorio de resultados)
    """
    print(f"\n{'='*100}")
    print(f"EXPERIMENTOS DE CLUSTERING CON IRIS")
    print(f"{'='*100}")
    
    # 1. Preparar directorio de resultados
    dirResultados = prepararDirectorioResultados()
    
    # 2. Cargar dataset
    datosOriginales, etiquetasOriginales = cargarDatasetIris()
    
    # 3. Ejecutar experimentos con diferentes semillas para ambos métodos
    # Ejemplo: numSemillas=4, semillaInicial=42 genera semillas 42, 43, 44, 45
    resultadosCluORT, resultadosKMeans = ejecutarExperimentos(datosOriginales, etiquetasOriginales, numSemillas=1, semillaInicial=0)
    
    # 4. Crear tabla resumen
    
    dfResultados = crearTablaResumen(resultadosCluORT, resultadosKMeans, dirResultados)
    """"
    # 5. Generar gráficos
    
    print(f"\n{'='*100}")
    print(f"GENERANDO GRÁFICOS")
    print(f"{'='*100}")
    
    # Evolución de métricas
    graficarEvolucionMetricas(dfResultados, dirResultados)
    graficarEvolucionMetricasCluORT(dfResultados, dirResultados)
    graficarEvolucionMetricasSTREAMKMeans(dfResultados, dirResultados)
    
    # Gráfico comparativo: CluORT vs STREAMKMeans lado a lado con métricas
    print("Generando gráfico comparativo CluORT vs STREAMKMeans...")
    graficarComparativaAmbosMetodos(resultadosCluORT, resultadosKMeans, datosOriginales, etiquetasOriginales, dirResultados)
    
    # Gráfico de distribución conjunta
    print("Generando gráfico de distribución conjunta...")
    graficarDistribucionConjunta(resultadosCluORT, resultadosKMeans, datosOriginales, etiquetasOriginales, dirResultados)
    
    # Gráfico de distribución para CluORT
    print("Generando gráfico de distribución CluORT separada...")
    graficarRestriccionesTamanio(resultadosCluORT, datosOriginales, etiquetasOriginales, dirResultados, sufijo='_cluort')
    
    # Gráfico de distribución para STREAMKMeans
    print("Generando gráfico de distribución STREAMKMeans separada...")
    graficarRestriccionesTamanio(resultadosKMeans, datosOriginales, etiquetasOriginales, dirResultados, sufijo='_streamkmeans')
    
    print(f"\n{'='*100}")
    print(f"EXPERIMENTOS COMPLETADOS")
    print(f"{'='*100}")
    print(f"Resultados guardados en: {dirResultados}")
    print(f"{'='*100}\n")
    """


if __name__ == "__main__":
    main()

    
