import pandas as pd
import numpy as np
import sqlite3
import os

from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor ,LGBMRanker
from sklearn.metrics import  ndcg_score
import joblib
import json
import time

from src.querys import query_caso_1,query_caso_2,query_caso_3
from src.config import BASE_DB,TABLA_MODELO_FULL,EJEMPLO_FILE , OUTPUTS_MODELO_FINAL,OUTPUTS_DATASETS
from src.config import FEATURES ,CAT_FEATURES,TARGET
from src.feat_eng import pipeline_feature_engineering
from src.retrievals import obtener_libros_leidos,retrieval_q1_populares,retrieval_q2_mismo_autor,retrieval_q3_mismo_genero,retrieval_q4_popularidad_pais,retrieval_q5_trending,retrieval_hibrido

def train_test_split_dfs(tabla = TABLA_MODELO_FULL): 
    print(f"************************************ Comenzando split train test ************************************")
    train_file = "df_train.csv"
    test_file = "df_test.csv"

    exist_train = False
    exist_test = False 
    if os.path.exists(OUTPUTS_DATASETS + train_file):
        print("Ya existe archivo train")
        exist_train = True
    if os.path.exists(OUTPUTS_DATASETS + test_file):
        print("Ya existe archivo test")
        exist_test = True


    
    if exist_train and exist_test :
        print(f"exist train : {exist_train}   and   exist_test :{exist_test}")
        df_train = pd.read_csv(OUTPUTS_DATASETS + train_file)
        df_test = pd.read_csv(OUTPUTS_DATASETS + test_file)
        return df_train , df_test
    else : 
        con = sqlite3.connect(BASE_DB)
        id_lectores = pd.read_sql(f"""
                        SELECT id_lector
                        FROM {tabla}
                        GROUP BY id_lector
                        HAVING COUNT(*) >= 20
                    """, con)["id_lector"]
        df_train = pd.DataFrame({"id_lector": [], "id_libro": [], "rating": []})
        df_test = pd.DataFrame({"id_lector": [], "id_libro": [], "rating": []})

        for id_lector in id_lectores:
            id_libros = pd.read_sql(f"""
                SELECT id_lector, id_libro, rating
                FROM {tabla}
                WHERE id_lector = ?
                ORDER BY fecha
            """, con, params=[id_lector])


            df_train = pd.concat([df_train, id_libros[:-20]], axis=0)
            df_test = pd.concat([df_test, id_libros[-20:]], axis=0)

        con.close()
        # Guardo los archivos
        df_train.to_csv(OUTPUTS_DATASETS + train_file, index=False)
        df_test.to_csv(OUTPUTS_DATASETS + test_file, index=False)

    return df_train , df_test



def todos_los_libros()-> pd.DataFrame:
    print(f"************************************ Comenzando lista todos los libros  ************************************")

    
    con = sqlite3.connect(BASE_DB)

    list_todos_los_libros = pd.read_sql("SELECT id_libro FROM libros", con)["id_libro"]
    con.close()
    return list_todos_los_libros

#def retrieval(df_train:pd.DataFrame, id_lector  , list_todos_los_libros:list):
 #   """Retorna todos los libros que se pueden recomendar a id_lector"""
  #  print(f"************************************ Comenzando retrieval libros para id : {id_lector}************************************")


#    libros_leidos = df_train[df_train["id_lector"] == id_lector]["id_libro"].to_list()
 #   libros_no_leidos = list_todos_los_libros[~list_todos_los_libros.isin(libros_leidos)]

  #  print(f"Libros leidos : {len(libros_leidos)}")
   # print(f"Libros no leidos : {len(libros_no_leidos)}")

    #return libros_no_leidos.to_list()


def selecc_features(feats_outs_exp :list[str]):
    print("************************************ Comienzo seleccion features ************************************")
    features_seleccionadas = [c for c in FEATURES if c not in feats_outs_exp]
    print(f"Features Seleccionadas : {features_seleccionadas}")
    return features_seleccionadas


def model_baseline_train(df_train:pd.DataFrame ,features_seleccionadas:list[str] , seed:int):
    print("************************************ Comienzo entrenamiendo modelo base ************************************")
    print(f"Features seleccionadas para el entrenamiento : {features_seleccionadas}")
    print(f"Target : {TARGET}")

    cat_features=[c for c in CAT_FEATURES if c in features_seleccionadas]

    for c in cat_features:
            df_train[c] = df_train[c].astype('category')

    modelo_lgbm = LGBMRegressor(
        random_state=seed
    )
    X_train = df_train[features_seleccionadas]
    y_train = df_train[TARGET]

    modelo_lgbm.fit(
        X_train, 
        y_train,
        categorical_feature=cat_features 
    ) 
    return modelo_lgbm

from lightgbm import LGBMRanker # Asegurate de tener esta importación arriba de todo

def model_baseline_train_ranker(df_train: pd.DataFrame, features_seleccionadas: list[str], seed: int):
    print("************************************ Comienzo entrenamiento modelo RANKER (LGBMRanker) ************************************")
    print(f"Features seleccionadas para el entrenamiento : {features_seleccionadas}")
    print(f"Target : {TARGET}")

    # 1. ORDENAR POR USUARIO (CRÍTICO PARA RANKING)
    # LightGBM exige que todas las interacciones de un mismo query (usuario) estén juntas contiguamente.
    df_train = df_train.sort_values(by="id_lector").reset_index(drop=True)

    # 2. ARMAR EL ARREGLO DE GRUPOS
    # Contamos exactamente cuántos libros tiene cada usuario en este set de entrenamiento.
    # sort=False es vital porque ya ordenamos el DataFrame arriba.
    group_counts = df_train.groupby("id_lector", sort=False).size().values

    cat_features = [c for c in CAT_FEATURES if c in features_seleccionadas]

    for c in cat_features:
        df_train[c] = df_train[c].astype('category')

    # 3. INICIALIZAR EL MODELO RANKER
    # lambdarank es la función de pérdida matemática estándar para optimizar NDCG
    modelo_lgbm = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        random_state=seed,
        importance_type="gain"
    )
    
    X_train = df_train[features_seleccionadas]
    y_train = df_train[TARGET]

    # 4. ENTRENAR PASANDO EL PARÁMETRO GROUP
    modelo_lgbm.fit(
        X_train, 
        y_train,
        group=group_counts,
        categorical_feature=cat_features 
    ) 
    
    return modelo_lgbm
import os
import time
import json
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

def evaluar_experimentos(df_test, df_train_features, modelo_lgbm=None):
    
    # 1. Definimos el diccionario de Retrievals a evaluar
    estrategias_retrieval = {
        "Q1_Populares": lambda u: retrieval_q1_populares(df_train_features, u, top_n=5000),
        "Q2_Mismo_Autor" : lambda u: retrieval_q2_mismo_autor(df_train_features, u, top_n=5000),
        "Q3_Mismo_Genero" : lambda u: retrieval_q3_mismo_genero(df_train_features, u, top_n=5000),
        "Q4_Popularidad_Pais": lambda u: retrieval_q4_popularidad_pais(df_train_features, u, top_n=5000),
        "Q5_Trending": lambda u: retrieval_q5_trending(df_train_features, u, top_n=5000),
        "Hibrido": lambda u: retrieval_hibrido(df_train_features, u, top_n_por_query=50)
    }

    # 2. Definimos los Rankers
    estrategias_ranking = ["Rank_Popularidad"]
    if modelo_lgbm is not None:
        estrategias_ranking.append("Rank_LGBMRanker")
        features_del_modelo = modelo_lgbm.feature_name_
    
    pop_dict = df_train_features.groupby("id_libro")["rating_promedio_te_por_libro"].max().to_dict()
    media_global = df_train_features["rating"].mean()
    
    resultados_totales = []

    # Directorio para guardar checkpoints
    os.makedirs("outputs/checkpoints", exist_ok=True)
    
    # Iteramos sobre la matriz de experimentos (Retrieval x Ranker)
    for nombre_ret, func_retrieval in estrategias_retrieval.items():
        for nombre_rank in estrategias_ranking:
            
            print(f"\n ***************** Evaluando: Retrieval=[{nombre_ret}] | Ranker=[{nombre_rank}]***************")
            inicio = time.time()

            checkpoint_file = f"outputs/checkpoints/ckpt_{nombre_ret}_{nombre_rank}.csv"
            usuarios_procesados = {}
            
            # --- RECUPERA CHECKPOINT (Si el script se había cortado) ---
            if os.path.exists(checkpoint_file):
                print(f" -> Encontrado checkpoint previo: {checkpoint_file}. Retomando...")
                df_ckpt = pd.read_csv(checkpoint_file)
                for _, row in df_ckpt.iterrows():
                    usuarios_procesados[row["id_lector"]] = {
                        "recall": row["recall"],
                        "ndcg": row["ndcg"]
                    }

            # Cargamos las listas con los valores ya calculados de antes
            ndcg_lista = [v["ndcg"] for v in usuarios_procesados.values()]
            recall_retrieval_lista = [v["recall"] for v in usuarios_procesados.values()]

            usuarios_test = df_test["id_lector"].unique() 
            nuevos_procesados_en_esta_sesion = 0
            
            # --- BUCLE DE EVALUACIÓN POR USUARIO ---
            for id_lector in usuarios_test:
                
                # ¡MAGIA!: Si el usuario ya está en el checkpoint, lo salta al instante
                if id_lector in usuarios_procesados:
                    continue
                    
                # --- FASE 1: RETRIEVAL ---
                libros_candidatos = func_retrieval(id_lector)
                
                subset_test = df_test[df_test["id_lector"] == id_lector]
                true_relevance_dict = subset_test.set_index("id_libro")["rating"].to_dict()
                libros_test_reales = set(true_relevance_dict.keys())
                
                hits = len(set(libros_candidatos).intersection(libros_test_reales))
                recall_retrieval = hits / len(libros_test_reales) if len(libros_test_reales) > 0 else 0

                # --- FASE 2: RANKING ---
                predicted_scores_dict = {}

                
                if len(libros_candidatos) > 0:
                    if nombre_rank == "Rank_Popularidad":
                        for lib in libros_candidatos:
                            predicted_scores_dict[lib] = pop_dict.get(lib, media_global)
                            
                    elif nombre_rank == "Rank_LightGBM" and modelo_lgbm is not None:
                        df_candidatos = pd.DataFrame({
                                "id_lector": [id_lector] * len(libros_candidatos),
                                "id_libro": libros_candidatos
                            })
                        
                        df_candidatos_feat = pipeline_feature_engineering(df_candidatos, state="test", df_historia=df_train_features)
                        
                        X_candidatos = df_candidatos_feat[features_del_modelo].copy()

                        cat_features_eval = [c for c in CAT_FEATURES if c in features_del_modelo]
                        for c in cat_features_eval:
                            # Alineamos categorías con el train para que LightGBM no explote
                            X_candidatos[c] = pd.Categorical(X_candidatos[c], categories=df_train_features[c].cat.categories)


                        y_pred = modelo_lgbm.predict(X_candidatos)
                        predicted_scores_dict = dict(zip(libros_candidatos, y_pred))

                # --- FASE 3: MÉTRICAS DE RANKING (NDCG) ---
                # ¡EL PARCHE PARA NO INFLAR EL NDCG EN COLD-START!
                if len(libros_candidatos) == 0:
                    ndcg_val = 0.0
                else:
                    id_libros_totales = list(set(list(true_relevance_dict.keys()) + list(predicted_scores_dict.keys())))
                    
                    if len(id_libros_totales) > 0:
                        y_true = np.asarray([[true_relevance_dict.get(id_libro, 0) for id_libro in id_libros_totales]])
                        #y_score = np.asarray([[predicted_scores_dict.get(id_libro, 0) for id_libro in id_libros_totales]])
                        y_score = np.asarray([[predicted_scores_dict.get(id_libro, -99999) for id_libro in id_libros_totales]])
                        
                        ndcg_val = ndcg_score(y_true, y_score, k=20)
                    else:
                        ndcg_val = 0.0

                ndcg_lista.append(ndcg_val)
                recall_retrieval_lista.append(recall_retrieval)
                # --- GUARDA CHECKPOINT (Después de cada usuario) ---
                # Creamos un DF con 1 sola fila
                registro_nuevo = pd.DataFrame([{
                    "id_lector": id_lector,
                    "recall": recall_retrieval,
                    "ndcg": ndcg_val
                }])
                # Se pega (append) al archivo en disco sin borrar lo anterior
                registro_nuevo.to_csv(checkpoint_file, mode='a', header=not os.path.exists(checkpoint_file), index=False)
                
                nuevos_procesados_en_esta_sesion += 1
                if nuevos_procesados_en_esta_sesion % 50 == 0:
                    print(f"   -> Procesados {nuevos_procesados_en_esta_sesion} usuarios...")

            # --- FIN DEL BUCLE DE USUARIOS (Cálculo promedio del experimento) ---
            resultados_totales.append({
                "Retrieval": nombre_ret,
                "Ranker": nombre_rank,
                "Recall_Retrieval_Promedio": np.mean(recall_retrieval_lista),
                "NDCG@20_Promedio": np.mean(ndcg_lista),
                "Tiempo_Segundos": round(time.time() - inicio, 2)
            })
            
            print(f" -> NDCG@20: {np.mean(ndcg_lista):.4f} | Recall: {np.mean(recall_retrieval_lista):.4f}")

    # Al finalizar todos los cruces, exporta la tabla bonita
    df_resultados = pd.DataFrame(resultados_totales)
    df_resultados.to_csv("outputs/datasets/resultados_experimentos.csv", index=False)
    
    return df_resultados






























import pandas as pd

def calcular_feature_importance(modelo_lgbm,experiment_name :str ,is_modelo_final = False) -> pd.DataFrame:
    """
    Extrae la importancia de las variables de un modelo LightGBM entrenado,
    calculando por Split, por Gain y el porcentaje relativo del Gain.
    """
    print("************************************ Calculando Feature Importances ************************************")
    
    # 1. Extraemos los nombres de las variables directamente del modelo
    features = modelo_lgbm.feature_name_
    
    # 2. Extraemos ambas métricas desde el 'booster' interno de LightGBM
    importancia_split = modelo_lgbm.booster_.feature_importance(importance_type='split')
    importancia_gain = modelo_lgbm.booster_.feature_importance(importance_type='gain')
    
    # 3. Armamos el DataFrame
    df_importances = pd.DataFrame({
        'Feature': features,
        'Split': importancia_split,
        'Gain': importancia_gain
    })
    
    # 4. Calculamos el porcentaje sobre la ganancia total (Gain)
    total_gain = df_importances['Gain'].sum()
    df_importances['Gain_Porcentaje'] = (df_importances['Gain'] / total_gain) * 100
    
    # 5. Ordenamos por el porcentaje de Gain (de mayor a menor)
    df_importances = df_importances.sort_values(by='Gain_Porcentaje', ascending=False).reset_index(drop=True)
    if is_modelo_final:
        file_name = OUTPUTS_MODELO_FINAL + f"feat_imp_{experiment_name}_final.csv"
        df_importances.to_csv(file_name)
        print(f"feat importances guardado en : {file_name}")
    
    return df_importances



def entrenamiento_modelo_final(df:pd.DataFrame ,features_seleccionadas:list[str] , seed:int,experiment_name :str):
    print(f"************************************ Comienzo entrenamiendo modelo base final :{experiment_name} ************************************")
    print(f"Features seleccionadas para el entrenamiento : {features_seleccionadas}")
    print(f"Target : {TARGET}")

    cat_features=[c for c in CAT_FEATURES if c in features_seleccionadas]

    modelo_lgbm = LGBMRegressor(
        random_state=seed
    )
    X = df[features_seleccionadas]
    y = df[TARGET]

    modelo_lgbm.fit(
        X, 
        y,
        categorical_feature=cat_features 
    ) 
    file_model = OUTPUTS_MODELO_FINAL + f"modelo_final_{experiment_name}.pkl"

    joblib.dump(modelo_lgbm, file_model)
    print(f"Modelo guardado exitosamente en: {file_model}")

    feat_imp_final = calcular_feature_importance(modelo_lgbm,experiment_name , True)
    return modelo_lgbm ,feat_imp_final

def prediccion_usuarios_final(df_base_modelado:pd.DataFrame,features_seleccionadas:list[str] , model,experiment_name :str):
    """
    df_base_modelado es el df que viene del flujo con que se modelo --> De la tabla BASE_MODELO_FULL_FEAT_ENG. 
    Lo usamos para poder identificar a los ids que ya estaban ahi.  Los que no estaban eran :
            * alpacil : estaba en interacciones pero no en lectores
            * '102893014085601872783', '103084137542939876557', '104186425412399255991', 
            '114281238546612894708', '115834876939320404006', '115884316008551273059',
                '116044910452339432768', 'acastrop', 'alpasil', 'carlosecheve', 'diegodacosta',
                  'eduardexhp', 'india', 'pequod', 'raf86', 'tommycaceres','victor27' : No estaban en interacciones
    """
    print("************************************ Comienzo prediccion final ************************************")

    cat_features=[c for c in CAT_FEATURES if c in features_seleccionadas]

    df_ejemplo = pd.read_csv(EJEMPLO_FILE)
    ids_a_predecir=list(df_ejemplo.id_lector.unique())
    ids_modelado = list(df_base_modelado.id_lector.unique())

    # Me traigo todos los libros
    con = sqlite3.connect(BASE_DB)


    query = """SELECT distinct(id_lector) FROM interacciones i left join lectores l using(id_lector) where l.id_lector IS NULL """
    ids_si_inter_no_lectores = pd.read_sql_query(query,con)
    ids_si_inter_no_lectores=ids_si_inter_no_lectores.id_lector.to_list()
    ids_si_inter_no_lectores = [c for c in ids_si_inter_no_lectores if c in ids_a_predecir]

    query = """SELECT distinct(id_lector) FROM lectores l left join interacciones i using(id_lector) where i.id_lector IS NULL """
    ids_no_inter_si_lectores = pd.read_sql_query(query,con)
    ids_no_inter_si_lectores=ids_no_inter_si_lectores.id_lector.to_list()
    ids_no_inter_si_lectores = [c for c in ids_no_inter_si_lectores if c in ids_a_predecir]


    #con.close()
    # Definimos las rutas de los archivos
    file_json_temp = OUTPUTS_MODELO_FINAL + f"checkpoint_{experiment_name}.json"
    try :
        with open(file_json_temp,'r',encoding='utf-8') as f:
           resultados_dict=json.load(f)
           if len(resultados_dict.keys()) == len(ids_a_predecir):
               print(f"Ya esta completado el json de los resultados : {file_json_temp} !! Pasar a la preparacion del envio")
               return
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Aun no se comenzo con los resultados finales del exp : {experiment_name}")
        resultados_dict = {}

    for id_i in ids_a_predecir:
        if id_i in resultados_dict.keys():
            continue

        if id_i in ids_modelado:
            query=query_caso_1
            df_candidatos = pd.read_sql_query(query, con, params=(id_i, id_i, id_i))
            for col in cat_features:
                if col in df_candidatos.columns:
                    df_candidatos[col] = df_candidatos[col].astype('category')
                    
            # 2. Forzamos a numéricas (float) el resto de las variables seleccionadas
            num_features = [c for c in features_seleccionadas if c not in cat_features]
            for col in num_features:
                if col in df_candidatos.columns:
                    df_candidatos[col] = pd.to_numeric(df_candidatos[col], errors='coerce').astype(float)
            X = df_candidatos[features_seleccionadas]
            df_candidatos['score'] = model.predict(X)

            # NUEVO: Guardamos ID y Score
            top_20_df = df_candidatos.sort_values(by='score', ascending=False).head(20).copy()
            top_20_df['score'] = top_20_df['score'].astype(float) # Casteo seguro para JSON
            top_20_libros = top_20_df[['id_libro', 'score']].to_dict(orient='records')
            
            resultados_dict[id_i] = top_20_libros
            with open(file_json_temp, 'w') as f:
                json.dump(resultados_dict, f)
            print(f"¡Proceso para el id {id_i} terminado. Respaldo JSON en: {file_json_temp}")

        elif id_i in ids_si_inter_no_lectores:
            print(f"el id:{id_i} entra a ids_si_inter_no_lectores")
            query = query_caso_2
            df_candidatos = pd.read_sql_query(query, con, params=(id_i, id_i, id_i, id_i, id_i, id_i))
            
            for col in cat_features:
                if col in df_candidatos.columns:
                    df_candidatos[col] = df_candidatos[col].astype('category')
                    
            # 2. Forzamos a numéricas (float) el resto de las variables seleccionadas
            num_features = [c for c in features_seleccionadas if c not in cat_features]
            for col in num_features:
                if col in df_candidatos.columns:
                    df_candidatos[col] = pd.to_numeric(df_candidatos[col], errors='coerce').astype(float)
            X = df_candidatos[features_seleccionadas]
            df_candidatos['score'] = model.predict(X)
            
            # NUEVO: Guardamos ID y Score
            top_20_df = df_candidatos.sort_values(by='score', ascending=False).head(20).copy()
            top_20_df['score'] = top_20_df['score'].astype(float) # Casteo seguro para JSON
            top_20_libros = top_20_df[['id_libro', 'score']].to_dict(orient='records')
            
            resultados_dict[id_i] = top_20_libros
            with open(file_json_temp, 'w') as f:
                json.dump(resultados_dict, f)
            print(f"¡Proceso para el id {id_i} terminado. Respaldo JSON en: {file_json_temp}")
        elif id_i in ids_no_inter_si_lectores:
            print(f"el id:{id_i} entra a ids_no_inter_si_lectores")
            query = query_caso_3
            df_candidatos = pd.read_sql_query(query, con, params=(id_i,))
            for col in cat_features:
                if col in df_candidatos.columns:
                    df_candidatos[col] = df_candidatos[col].astype('category')
                    
            # 2. Forzamos a numéricas (float) el resto de las variables seleccionadas
            num_features = [c for c in features_seleccionadas if c not in cat_features]
            for col in num_features:
                if col in df_candidatos.columns:
                    df_candidatos[col] = pd.to_numeric(df_candidatos[col], errors='coerce').astype(float)
            
            X = df_candidatos[features_seleccionadas]
            df_candidatos['score'] = model.predict(X)
            
            # NUEVO: Guardamos ID y Score
            top_20_df = df_candidatos.sort_values(by='score', ascending=False).head(20).copy()
            top_20_df['score'] = top_20_df['score'].astype(float) # Casteo seguro para JSON
            top_20_libros = top_20_df[['id_libro', 'score']].to_dict(orient='records')
            
            resultados_dict[id_i] = top_20_libros
            with open(file_json_temp, 'w') as f:
                json.dump(resultados_dict, f)
            print(f"¡Proceso para el id {id_i} terminado. Respaldo JSON en: {file_json_temp}")

    print(f"¡Proceso TOTAL terminado! Respaldo JSON en: {file_json_temp}")

    con.close()
    return resultados_dict
def preparacion_envio(experiment_name):
    # Definimos la ruta para el segundo archivo con scores
    file_results_csv = OUTPUTS_MODELO_FINAL + f"results_finales_{experiment_name}.csv"
    file_results_scores_csv = OUTPUTS_MODELO_FINAL + f"results_finales_{experiment_name}_con_scores.csv"
    file_json_temp = OUTPUTS_MODELO_FINAL + f"checkpoint_{experiment_name}.json"

    with open(file_json_temp,'r',encoding='utf-8') as f:
        resultados_dict=json.load(f)

    filas_kaggle = []
    filas_analisis = []
    
    for lector, libros in resultados_dict.items():
        # Ahora "libros" es una lista de diccionarios
        for item in libros:
            # Archivo para Kaggle (sin score)
            filas_kaggle.append({'id_lector': lector, 'id_libro': item['id_libro']})
            # Archivo para Análisis (con score)
            filas_analisis.append({'id_lector': lector, 'id_libro': item['id_libro'], 'score': item['score']})
            
    # 1. Armamos y exportamos el archivo oficial (Kaggle)
    resultados_finales = pd.DataFrame(filas_kaggle)
    resultados_finales.to_csv(file_results_csv, index=False)
    
    # 2. Armamos y exportamos el archivo interno (Análisis)
    resultados_con_scores = pd.DataFrame(filas_analisis)
    resultados_con_scores.to_csv(file_results_scores_csv, index=False)
    
    print(f"CSV para Kaggle generado en: {file_results_csv}")
    print(f"CSV de análisis con scores generado en: {file_results_scores_csv}")
    
    # Opcional: devolvemos ambos DataFrames por si querés usarlos en el Notebook
    return resultados_finales, resultados_con_scores

def preparacion_envio_old(experiment_name):
    file_results_csv = OUTPUTS_MODELO_FINAL + f"results_finales_{experiment_name}.csv"
    file_json_temp = OUTPUTS_MODELO_FINAL + f"checkpoint_{experiment_name}.json"

    with open(file_json_temp,'r',encoding='utf-8') as f:
        resultados_dict=json.load(f)

    filas_para_df = []
    for lector, libros in resultados_dict.items():
        for libro in libros:
            filas_para_df.append({'id_lector': lector, 'id_libro': libro})
            
    resultados_finales = pd.DataFrame(filas_para_df)
    resultados_finales.to_csv(file_results_csv, index=False)
    print(f"CSV final generado en: {file_results_csv}")
    
    return resultados_finales