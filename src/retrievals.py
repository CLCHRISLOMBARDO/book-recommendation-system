import pandas as pd
import sqlite3
from src.config import BASE_DB
## Los df pueden ser : 
        # df_train
        # df_total
        # df del segmento especifico

def obtener_libros_leidos(df: pd.DataFrame, id_lector):
    """Devuelve una lista con los IDs de los libros que el usuario ya leyó en train."""
    return df[df["id_lector"] == id_lector]["id_libro"].tolist()


def retrieval_q1_populares(df: pd.DataFrame, id_lector, top_n=100) -> list:
    """Q1: Los libros más populares (Se calcula sobre df_train para no espiar test)."""
    libros_leidos = obtener_libros_leidos(df, id_lector)
    
    # Agrupamos en el df_train para saber cuáles son los más leídos de la historia
    populares = df["id_libro"].value_counts().index.tolist()
    
    # Filtramos los que el usuario ya leyó
    candidatos = [libro for libro in populares if libro not in libros_leidos][:top_n]
    return candidatos


def retrieval_q2_mismo_autor(df: pd.DataFrame, id_lector, top_n=100) -> list:
    """Q2: Libros de los mismos autores que el usuario ya leyó usando SQL."""
    libros_leidos = obtener_libros_leidos(df, id_lector)
    
    # 1. Buscamos qué autores leyó en TRAIN (esquivando Data Leakage)
    autores_leidos = df[df["id_lector"] == id_lector]["autor"].dropna().unique().tolist()
    
    if len(autores_leidos) == 0:
        return [] # Usuario Cold-Start
        
    # 2. Vamos a la base SQL a buscar TODOS los libros de esos autores
    con = sqlite3.connect(BASE_DB)
    # Armamos la cantidad de signos de interrogación para la query ('?', '?', '?')
    placeholders = ','.join(['?'] * len(autores_leidos)) 
    query = f"SELECT id_libro FROM libros WHERE autor IN ({placeholders})"
    
    libros_del_autor = pd.read_sql(query, con, params=autores_leidos)["id_libro"].tolist()
    con.close()
    
    # 3. Filtramos los que ya leyó
    candidatos = [libro for libro in libros_del_autor if libro not in libros_leidos][:top_n]
    return candidatos


def retrieval_q3_mismo_genero(df: pd.DataFrame, id_lector, top_n=100) -> list:
    """Q3: Libros de los mismos géneros que el usuario ya leyó usando SQL."""
    libros_leidos = obtener_libros_leidos(df, id_lector)
    
    # 1. Buscamos qué géneros leyó en TRAIN
    # Nota: uso 'genero_libro' que es como nombraste a la columna en tu merge inicial
    generos_leidos = df[df["id_lector"] == id_lector]["genero_libro"].dropna().unique().tolist()
    
    if len(generos_leidos) == 0:
        return []
        
    # 2. Vamos a la base SQL a buscar TODOS los libros de esos géneros
    con = sqlite3.connect(BASE_DB)
    placeholders = ','.join(['?'] * len(generos_leidos))
    # Asumimos que en la tabla libros nativa la columna se llama 'genero'
    query = f"SELECT id_libro FROM libros WHERE genero IN ({placeholders})"
    
    libros_del_genero = pd.read_sql(query, con, params=generos_leidos)["id_libro"].tolist()
    con.close()
    
    # 3. Filtramos los que ya leyó
    candidatos = [libro for libro in libros_del_genero if libro not in libros_leidos][:top_n]
    return candidatos


def retrieval_q4_popularidad_pais(df: pd.DataFrame, id_lector, top_n=100) -> list:
    """Q5: Libros más populares en el país del lector. Excelente para Cold-Start."""
    libros_leidos = obtener_libros_leidos(df, id_lector)
    
    # 1. Buscamos el país del lector
    datos_lector = df[df["id_lector"] == id_lector]
    if datos_lector.empty:
        return []
        
    pais_lector = datos_lector["pais"].iloc[0]
    
    if pais_lector == "Desconocido":
        return [] # Si no sabemos el país, no podemos aportar nada
        
    # 2. Filtramos la base para quedarnos solo con interacciones de ese país
    df_pais = df[df["pais"] == pais_lector]
    
    # 3. Calculamos populares de ese país y filtramos los ya leídos
    populares_pais = df_pais["id_libro"].value_counts().index.tolist()
    candidatos = [libro for libro in populares_pais if libro not in libros_leidos][:top_n]
    
    return candidatos


def retrieval_q5_trending(df: pd.DataFrame, id_lector, top_n=100) -> list:
    """Q6: Libros trending (Más leídos en el último tiempo)."""
    libros_leidos = obtener_libros_leidos(df, id_lector)
    
    # Nos aseguramos de tener la fecha como datetime
    if "fecha_interaccion" not in df.columns:
        return []
        
    #df["fecha_interaccion"] = pd.to_datetime(df["fecha_interaccion"], format="%d-%m-%Y",errors='coerce')
    df["fecha_interaccion"] = pd.to_datetime(df["fecha_interaccion"], format="%d-%m-%Y")

    
    # Calculamos la fecha de corte (ej. el último año de interacciones en la base)
    fecha_maxima = df["fecha_interaccion"].max()
    fecha_corte = fecha_maxima - pd.DateOffset(months=12) 
    
    df_reciente = df[df["fecha_interaccion"] >= fecha_corte]
    populares_recientes = df_reciente["id_libro"].value_counts().index.tolist()
    
    candidatos = [libro for libro in populares_recientes if libro not in libros_leidos][:top_n]
    return candidatos


def retrieval_hibrido(df: pd.DataFrame, id_lector, top_n_por_query=50) -> list:
    """Unión de todas las estrategias."""
    
    c1 = retrieval_q1_populares(df, id_lector, top_n=top_n_por_query)
    c2 = retrieval_q2_mismo_autor(df, id_lector, top_n=top_n_por_query)
    c3 = retrieval_q3_mismo_genero(df, id_lector, top_n=top_n_por_query)
    c4 = retrieval_q4_popularidad_pais(df, id_lector, top_n=top_n_por_query)
    c5 = retrieval_q5_trending(df, id_lector, top_n=top_n_por_query)
    
    # El set elimina los duplicados de las distintas fuentes
    candidatos_unicos = list(set(c1 + c2 + c3 + c4 + c5))
    
    return candidatos_unicos