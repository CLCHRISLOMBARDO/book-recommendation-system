import pandas as pd
import sqlite3

from src.config import BASE_DB
def _fing_mergeo_features_bases(df: pd.DataFrame, state="entrenamiento"):
    con = sqlite3.connect(BASE_DB)
    
    # 1. Traemos los catálogos por separado
    libros = pd.read_sql_query("""
        SELECT id_libro, titulo, autor, genero as genero_libro, 
               editorial, CAST(anio_edicion AS integer) AS anio_edicion, resumen 
        FROM libros
    """, con)
    
    lectores = pd.read_sql_query("""
        SELECT id_lector, nombre, genero, cast(vive_en as text) as vive_en, 
               cast(nacimiento as INTEGER) as nacimiento 
        FROM lectores
    """, con)
    con.close()
    
    # 2. Hacemos el cruce de forma independiente (¡Clave para los candidatos!)
    df_mergeo = pd.merge(df, libros, on="id_libro", how="left")
    df_mergeo = pd.merge(df_mergeo, lectores, on="id_lector", how="left")
    
    # 3. Solo si estamos entrenando, vamos a buscar la fecha real de la interacción
    if state in ["entrenamiento", "entrenamiento_final"]:
        con = sqlite3.connect(BASE_DB)
        interacciones_fecha = pd.read_sql_query("SELECT id_lector, id_libro, fecha as fecha_interaccion FROM interacciones", con)
        con.close()
        df_mergeo = pd.merge(df_mergeo, interacciones_fecha, on=["id_lector", "id_libro"], how="left")
    else:
        # En inferencia (test), la fecha de interacción no existe aún. 
        # Podemos poner la fecha actual o dejarla nula según cómo la use tu modelo después.
        df_mergeo["fecha_interaccion"] = pd.NaT

    return df_mergeo

def _fing_pais_lector(df: pd.DataFrame,state="entrenamiento") -> pd.DataFrame:
    col = "pais"
    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    
    # 1. Diccionario para estandarizar el país final (traducir y unificar)
    estandarizacion_paises = {
        "mexico": "México",
        "méxico": "México",
        "united states of america": "Estados Unidos",
        "france, french republic": "Francia",
        "france": "Francia",
        "germany": "Alemania",
        "united kingdom": "Reino Unido",
        "italy": "Italia",
        "brazil": "Brasil",
        "cote d'ivoire": "Costa de Marfil",
        "portugal, portuguese republic": "Portugal",
        "switzerland, swiss confederation": "Suiza",
        "netherlands the": "Países Bajos",
        "netherlands antilles": "Antillas Neerlandesas",
        "dominican republic": "República Dominicana",
        "peru": "Perú",
        "panama": "Panamá",
        "romania": "Rumania"
    }

    # 2. Diccionario para ciudades huérfanas o países sueltos
    mapeo_directo = {
        "españa": "España",
        "espana": "España",
        "madrid": "España",
        "murcia": "España",
        "barcelona": "España",
        "málaga": "España",
        "valencia": "España",
        "zaragoza": "España",
        "tarragona": "España",
        "tarazona": "España",
        "santander": "España",
        "torrejón de ardoz": "España",
        "argentina": "Argentina",
        "capital federal": "Argentina",
        "wilde": "Argentina",
        "chivilcoy": "Argentina",
        "mendoza": "Argentina",
        "el colorado": "Argentina",
        "buenos aires": "Argentina",
        "chile": "Chile",
        "santiago": "Chile",
        "venezuela": "Venezuela",
        "san cristóbal": "Venezuela",
        "caracas": "Venezuela",
        "méxico": "México",
        "cdmx": "México",
        "culiacán": "México",
        "zapopan": "México",
        "mexicali": "México",
        "colombia": "Colombia",
        "bogotá": "Colombia",
        "bogota": "Colombia",
        "cali": "Colombia",
        "cucuta": "Colombia",
        "guatemala": "Guatemala",
        "costa rica": "Costa Rica",
        "heredia": "Costa Rica",
        "san josé": "Costa Rica",
        "cartago": "Costa Rica",
        "san ramón, alajuela": "Costa Rica",
        "punta cana": "República Dominicana",
        "santo domingo este": "República Dominicana",
        "london": "Reino Unido",
        "surbiton": "Reino Unido",
        "berlín": "Alemania",
        "södertälje": "Suecia"
    }

    def extraer_pais(texto):
        if pd.isna(texto):
            return "Desconocido"
            
        t = str(texto).lower().strip()
        
        # Limpiamos terminaciones sucias ("Madrid - ¿?", "Santiago -")
        if t.endswith("- ¿?") or t.endswith("-"):
            t = t.replace("- ¿?", "").replace("-", "").strip()
            
        # Si quedó la ciudad sola o era un país suelto
        if t in mapeo_directo:
            return mapeo_directo[t]
            
        # Si tiene la estructura normal "Ciudad - País"
        if "-" in t:
            partes = [p.strip() for p in t.split("-")]
            posible_pais = partes[-1]
        else:
            posible_pais = t

        # Estandarizamos el país si está en el diccionario de traducción
        if posible_pais in estandarizacion_paises:
            return estandarizacion_paises[posible_pais]
            
        return posible_pais.title()

    df[col] = df["vive_en"].apply(extraer_pais)
    return df

def _fing_volumen_lectura_por_lector(df: pd.DataFrame, state="entrenamiento", df_historia=None) -> pd.DataFrame:
    col = "volumen_lectura_por_lector"
    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    if state in ["entrenamiento","entrenamiento_final"]:
        #Este estado puede ser el entrenamiento inicial (df train) o el final para enviar a kaggle (df total)
        df[col] = df.groupby("id_lector")["id_libro"].transform("count")

    elif state in ["test", "prediccion_final"]:
        # 1. Si no nos pasan la historia por memoria, hacemos el fallback al disco (por si acaso)
        if df_historia is None:
            file_name = "df_train_features.csv" if state == "test" else "df_total_features.csv"
            df_historia = pd.read_csv(f"outputs/datasets/{file_name}")
            
        # 2. Hacemos el merge directo con el dataframe en memoria
        df = pd.merge(df, df_historia[['id_lector', col]].drop_duplicates(subset=["id_lector"]), on="id_lector", how='left')
    
    df[col] = df[col].fillna(0)
    return df


def _fing_rating_promedio_crudo_por_lector(df: pd.DataFrame, state="entrenamiento", df_historia=None) -> pd.DataFrame:
    col = "rating_promedio_crudo_por_lector"

    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    if state in ["entrenamiento","entrenamiento_final"]:
            #Este estado puede ser el entrenamiento inicial (df train) o el final para enviar a kaggle (df total)
        df[col] = df.groupby("id_lector")["rating"].transform("mean")

    elif state in ["test", "prediccion_final"]:
        # 1. Si no nos pasan la historia por memoria, hacemos el fallback al disco (por si acaso)
        if df_historia is None:
            file_name = "df_train_features.csv" if state == "test" else "df_total_features.csv"
            df_historia = pd.read_csv(f"outputs/datasets/{file_name}")
            
        # 2. Hacemos el merge directo con el dataframe en memoria
        df = pd.merge(df, df_historia[['id_lector', col]].drop_duplicates(subset=["id_lector"]), on="id_lector", how='left')
    return df


def _fing_desviacion_estandar_rating_por_lector(df: pd.DataFrame, state="entrenamiento", df_historia=None) -> pd.DataFrame:
    col = "desviacion_estandar_rating_por_lector"

    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    if state in ["entrenamiento","entrenamiento_final"]:
    # ddof=0 para alinear con la fórmula poblacional de SQL
        std = df.groupby("id_lector")["rating"].transform(lambda x: x.std(ddof=0))
        df[col] = std.fillna(0.0).round(4)

    elif state in ["test", "prediccion_final"]:
        # 1. Si no nos pasan la historia por memoria, hacemos el fallback al disco (por si acaso)
        if df_historia is None:
            file_name = "df_train_features.csv" if state == "test" else "df_total_features.csv"
            df_historia = pd.read_csv(f"outputs/datasets/{file_name}")
            
        # 2. Hacemos el merge directo con el dataframe en memoria
        df = pd.merge(df, df_historia[['id_lector', col]].drop_duplicates(subset=["id_lector"]), on="id_lector", how='left')  

    df[col] = df[col].fillna(0)
    
    return df


def _fing_genero_favorito_por_lector(df: pd.DataFrame,state ="entrenamiento", df_historia=None) -> pd.DataFrame:
    col = "genero_favorito_por_lector"

    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    if state in ["entrenamiento","entrenamiento_final"]:
        # Filtrar nulos o vacíos
        validos = df[df["genero_libro"].notna() & (df["genero_libro"].astype(str).str.strip() != "")]
        # Moda por usuario
        top_genero = (
            validos.groupby("id_lector")["genero_libro"]
            .agg(lambda x: x.mode().iloc[0] if not x.empty else "DESCONOCIDO")
            .reset_index(name=col)
        )
        df = df.merge(top_genero, on="id_lector", how="left")
        df[col] = df[col].fillna("DESCONOCIDO")

    elif state in ["test", "prediccion_final"]:
        # 1. Si no nos pasan la historia por memoria, hacemos el fallback al disco (por si acaso)
        if df_historia is None:
            file_name = "df_train_features.csv" if state == "test" else "df_total_features.csv"
            df_historia = pd.read_csv(f"outputs/datasets/{file_name}")
            
        # 2. Hacemos el merge directo con el dataframe en memoria
        df = pd.merge(df, df_historia[['id_lector', col]].drop_duplicates(subset=["id_lector"]), on="id_lector", how='left') 

    return df


def _fing_antiguedad_lectura_por_lector(df: pd.DataFrame ,state = "entrenamiento", df_historia=None) -> pd.DataFrame:
    col="ratio_libros_antiguos_vs_nuevos_por_lector"
    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")

    if state in ["entrenamiento","entrenamiento_final"]:
        es_antiguo = (df["anio_edicion"] < 2000).astype(int)
        es_nuevo = (df["anio_edicion"] >= 2000).astype(int)

        stats_anio = df[["id_lector"]].copy()
        stats_anio["antiguo"] = es_antiguo
        stats_anio["nuevo"] = es_nuevo

        agrupado = stats_anio.groupby("id_lector").agg(
            cant_antiguos=("antiguo", "sum"),
            cant_nuevos=("nuevo", "sum"),
        )
        agrupado[col] = (
            agrupado["cant_antiguos"] / (agrupado["cant_nuevos"] + 1.0)
        ).round(4) 
        df = df.merge(agrupado[[col]], on="id_lector", how="left")

    elif state in ["test", "prediccion_final"]:
        # 1. Si no nos pasan la historia por memoria, hacemos el fallback al disco (por si acaso)
        if df_historia is None:
            file_name = "df_train_features.csv" if state == "test" else "df_total_features.csv"
            df_historia = pd.read_csv(f"outputs/datasets/{file_name}")
            
        # 2. Hacemos el merge directo con el dataframe en memoria
        df = pd.merge(df, df_historia[['id_lector', col]].drop_duplicates(subset=["id_lector"]), on="id_lector", how='left') 
    df[col] = df[col].fillna(0)
    return df


def _fing_rating_promedio_te_por_lector(df: pd.DataFrame, m_weight: float = 50.0,state="entrenamiento", df_historia=None) -> pd.DataFrame:
    col = "rating_promedio_te_por_lector"

    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    if state in ["entrenamiento","entrenamiento_final"]:
        mu = df["rating"].mean()
        
        stats = df.groupby("id_lector")["rating"].agg(n="count", mean_y="mean").reset_index()
        stats[col] = (
            (stats["n"] * stats["mean_y"] + m_weight * mu) / (stats["n"] + m_weight)
        ).round(4)

        df = df.merge(stats[["id_lector", col]], on="id_lector", how="left")

    elif state in ["test", "prediccion_final"]:
        # 1. Si no nos pasan la historia por memoria, hacemos el fallback al disco (por si acaso)
        if df_historia is None:
            file_name = "df_train_features.csv" if state == "test" else "df_total_features.csv"
            df_historia = pd.read_csv(f"outputs/datasets/{file_name}")
            
        # 2. Hacemos el merge directo con el dataframe en memoria
        mu = df_historia["rating"].mean()
        df = pd.merge(df, df_historia[['id_lector', col]].drop_duplicates(subset=["id_lector"]), on="id_lector", how='left') 
  
    df[col] = df[col].fillna(mu)
    return df

def _fing_pct_autor_favorito_por_lector(df: pd.DataFrame, state="entrenamiento", df_historia=None) -> pd.DataFrame:
    col = "pct_autor_favorito_por_lector"
    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    if state in ["entrenamiento","entrenamiento_final"]:

        # 2. Total de libros leídos por cada lector
        total_leidos = df.groupby("id_lector").size()
        
        # 3. Agrupar por (lector, autor), contar, y luego quedarnos con el máximo por lector
        max_por_autor = df.groupby(["id_lector", "autor"]).size().groupby(level="id_lector").max()
        
        # 4. Calcular el porcentaje y convertir la Serie a DataFrame
        df_pct = (max_por_autor / total_leidos).rename(col).reset_index()
        
        # 5. Pegar la nueva columna al df original (el que entró a la función)
        df = df.merge(df_pct, on="id_lector", how="left")

    elif state in ["test", "prediccion_final"]:
        # 1. Si no nos pasan la historia por memoria, hacemos el fallback al disco (por si acaso)
        if df_historia is None:
            file_name = "df_train_features.csv" if state == "test" else "df_total_features.csv"
            df_historia = pd.read_csv(f"outputs/datasets/{file_name}")
            
        # 2. Hacemos el merge directo con el dataframe en memoria
        df = pd.merge(df, df_historia[['id_lector', col]].drop_duplicates(subset=["id_lector"]), on="id_lector", how='left')    
    df[col] = df[col].fillna(0)
    return df

def _fing_pct_genero_favorito_por_lector(df: pd.DataFrame, state="entrenamiento", df_historia=None) -> pd.DataFrame:
    col = "pct_genero_favorito_por_lector"
    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    
    if state in ["entrenamiento", "entrenamiento_final"]:
        # 1. Total de libros leídos por cada lector
        total_leidos = df.groupby("id_lector").size()
        
        # 2. Agrupar por (lector, genero_libro), contar, y quedarnos con el máximo por lector
        max_por_genero = df.groupby(["id_lector", "genero_libro"]).size().groupby(level="id_lector").max()
        
        # 3. Calcular el porcentaje
        df_pct = (max_por_genero / total_leidos).rename(col).reset_index()
        
        # 4. Pegar al df original
        df = df.merge(df_pct, on="id_lector", how="left")

    elif state in ["test", "prediccion_final"]:
        # 1. Si no nos pasan la historia por memoria, hacemos el fallback al disco (por si acaso)
        if df_historia is None:
            file_name = "df_train_features.csv" if state == "test" else "df_total_features.csv"
            df_historia = pd.read_csv(f"outputs/datasets/{file_name}")
            
        # 2. Hacemos el merge directo con el dataframe en memoria
        df = pd.merge(df, df_historia[['id_lector', col]].drop_duplicates(subset=["id_lector"]), on="id_lector", how='left')  
    
    # Manejo de cold-start
    df[col] = df[col].fillna(0.0)
    
    return df

def _fing_popularidad_por_libro(df:pd.DataFrame,state ="entrenamiento", df_historia=None ) -> pd.DataFrame:
    col1 = "popularidad_count_por_libro"
    col2 = "popularidad_rating_mean_por_libro"
    if state != "test":
        print(f"-> Aplicando: {col1} y {col2} para el estado : {state}")

    if state in ["entrenamiento","entrenamiento_final"]:
        states_libro = df.groupby('id_libro').agg(**{
            col1: ('id_lector', 'count'),
            col2: ('rating', 'mean')
        })
        df = pd.merge(df , states_libro , on='id_libro' , how='left')

    elif state in ["test", "prediccion_final"]:
        # 1. Si no nos pasan la historia por memoria, hacemos el fallback al disco (por si acaso)
        if df_historia is None:
            file_name = "df_train_features.csv" if state == "test" else "df_total_features.csv"
            df_historia = pd.read_csv(f"outputs/datasets/{file_name}")
            
        # 2. Hacemos el merge directo con el dataframe en memoria
        df = pd.merge(df, df_historia[['id_libro', col1,col2]].drop_duplicates(subset=["id_libro"]), on="id_libro", how='left')  

    df[col1] = df[col1].fillna(0)
    df[col2] = df[col2].fillna(df[col2].mean()) # o el mu global
    return df 


def _fing_rating_promedio_te_por_libro(df: pd.DataFrame, m_weight: float = 50.0,state="entrenamiento", df_historia=None ) -> pd.DataFrame:
    col = "rating_promedio_te_por_libro"

    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    if state in ["entrenamiento","entrenamiento_final"]:
        mu = df["rating"].mean()
        
        stats = df.groupby("id_libro")["rating"].agg(n="count", mean_y="mean").reset_index()
        stats[col] = (
            (stats["n"] * stats["mean_y"] + m_weight * mu) / (stats["n"] + m_weight)
        ).round(4)

        df = df.merge(stats[["id_libro", col]], on="id_libro", how="left")

    elif state in ["test", "prediccion_final"]:
        # 1. Si no nos pasan la historia por memoria, hacemos el fallback al disco (por si acaso)
        if df_historia is None:
            file_name = "df_train_features.csv" if state == "test" else "df_total_features.csv"
            df_historia = pd.read_csv(f"outputs/datasets/{file_name}")
            
        # 2. Hacemos el merge directo con el dataframe en memoria
        mu = df_historia["rating"].mean()
        df = pd.merge(df, df_historia[['id_libro', col]].drop_duplicates(subset=["id_libro"]), on="id_libro", how='left') 

    df[col] = df[col].fillna(mu)
    return df


def _fing_pct_lecturas_autor_candidato_por_par_libroylector(df: pd.DataFrame, state="entrenamiento", df_historia=None ) -> pd.DataFrame:
    col = "pct_lecturas_autor_candidato_por_par_libroylector"
    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    
    if state in ["entrenamiento", "entrenamiento_final"]:
        # Usamos el propio df para calcular los historiales
        df_historia = df

    elif state in ["test", "prediccion_final"]:
        # 1. Si no nos pasan la historia por memoria, hacemos el fallback al disco (por si acaso)
        if df_historia is None:
            file_name = "df_train_features.csv" if state == "test" else "df_total_features.csv"
            df_historia = pd.read_csv(f"outputs/datasets/{file_name}")

    
    # 1. Calcular total de libros leídos por usuario en la historia
    total_leidos = df_historia.groupby("id_lector").size().rename("total_usuario")
    
    # 2. Calcular cuántos libros leyó de cada autor
    leidos_por_autor = df_historia.groupby(["id_lector", "autor"]).size().rename("leidos_del_autor")
    
    # 3. Unir y sacar el porcentaje
    stats_afinidad = pd.merge(leidos_por_autor, total_leidos, left_index=True, right_index=True).reset_index()
    stats_afinidad[col] = (stats_afinidad["leidos_del_autor"] / stats_afinidad["total_usuario"]).round(4)
    
    # 4. Mergear al df original (que tiene a los candidatos) usando AMBAS claves
    df = df.merge(stats_afinidad[["id_lector", "autor", col]], on=["id_lector", "autor"], how="left")
    
    # 5. Cold-Start o Sin Afinidad: Si nunca leyó a este autor, el porcentaje es 0
    df[col] = df[col].fillna(0.0)
    
    return df

def _fing_pct_lecturas_genero_candidato_por_par_libroylector(df: pd.DataFrame, state="entrenamiento", df_historia=None ) -> pd.DataFrame:
    col = "pct_lecturas_genero_candidato_por_par_libroylector"
    if state != "test":
        print(f"-> Aplicando: {col} para el estado : {state}")
    
    if state in ["entrenamiento", "entrenamiento_final"]:
        df_historia = df

    elif state in ["test", "prediccion_final"]:
        # 1. Si no nos pasan la historia por memoria, hacemos el fallback al disco (por si acaso)
        if df_historia is None:
            file_name = "df_train_features.csv" if state == "test" else "df_total_features.csv"
            df_historia = pd.read_csv(f"outputs/datasets/{file_name}")
    
    # 1. Calcular total de libros leídos por usuario en la historia
    total_leidos = df_historia.groupby("id_lector").size().rename("total_usuario")
    
    # 2. Calcular cuántos libros leyó de cada género
    leidos_por_genero = df_historia.groupby(["id_lector", "genero_libro"]).size().rename("leidos_del_genero")
    
    # 3. Unir y sacar el porcentaje
    stats_afinidad = pd.merge(leidos_por_genero, total_leidos, left_index=True, right_index=True).reset_index()
    stats_afinidad[col] = (stats_afinidad["leidos_del_genero"] / stats_afinidad["total_usuario"]).round(4)
    
    # 4. Mergear al df original usando AMBAS claves (Lector + Género)
    df = df.merge(stats_afinidad[["id_lector", "genero_libro", col]], on=["id_lector", "genero_libro"], how="left")
    
    # 5. Cold-Start o Sin Afinidad: Si nunca leyó este género, el porcentaje es 0
    df[col] = df[col].fillna(0.0)
    
    return df

def pipeline_feature_engineering(df: pd.DataFrame,state="entrenamiento" , df_historia=None) -> pd.DataFrame:
    """
    Ejecuta en cadena todas las funciones de Feature Engineering.
    """
    if state != "test":
        print("=" * 15 + f" INICIANDO PIPELINE DE FEATURE ENGINEERING del state : {state}" + "=" * 15)
    df_out = df.copy()

    # Cadena de transformaciones
    df_out = _fing_mergeo_features_bases(df_out, state )
    df_out = _fing_pais_lector(df_out,state)
    df_out = _fing_volumen_lectura_por_lector(df_out, state , df_historia=df_historia)
    df_out = _fing_rating_promedio_crudo_por_lector(df_out, state, df_historia=df_historia)
    df_out = _fing_desviacion_estandar_rating_por_lector(df_out, state, df_historia=df_historia)
    df_out = _fing_genero_favorito_por_lector(df_out, state, df_historia=df_historia)
    df_out = _fing_antiguedad_lectura_por_lector(df_out, state, df_historia=df_historia)
    df_out = _fing_rating_promedio_te_por_lector(df_out, state=state,m_weight=50.0, df_historia=df_historia)
    df_out = _fing_pct_autor_favorito_por_lector(df_out, state, df_historia=df_historia)
    df_out = _fing_pct_genero_favorito_por_lector(df_out, state, df_historia=df_historia)
    df_out = _fing_popularidad_por_libro(df_out,state, df_historia=df_historia)
    df_out = _fing_rating_promedio_te_por_libro(df_out, state=state,m_weight=50.0, df_historia=df_historia)
    df_out =_fing_pct_lecturas_autor_candidato_por_par_libroylector(df_out, state, df_historia=df_historia)
    df_out =_fing_pct_lecturas_genero_candidato_por_par_libroylector(df_out, state, df_historia=df_historia)


# Solo guardamos a disco si estamos entrenando
    if state in ["entrenamiento", "entrenamiento_final"]:
        file_name = "df_train_features.csv" if state == "entrenamiento" else "df_total_features.csv"
        print("=" * 15 + " GUARDANDO " + "=" * 15)
        df_out.to_csv(f"outputs/datasets/{file_name}", index=False)
    
    if state != "test":
        print("=" * 15 + " PIPELINE FINALIZADO CON ÉXITO " + "=" * 15)
    return df_out

