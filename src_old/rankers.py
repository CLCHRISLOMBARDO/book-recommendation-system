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
from src.config import BASE_DB,TABLA_MODELO_FULL,EJEMPLO_FILE ,OUTPUTS_MODELO_TRAIN, OUTPUTS_MODELO_FINAL,OUTPUTS_DATASETS ,OUTPUTS_CHECKPOINTS ,OUTPUTS_RESULTADOS
from src.config import FEATURES ,CAT_FEATURES,TARGET
from src.feat_eng import pipeline_feature_engineering
from src.retrievals import retrieval_q1_populares,retrieval_q2_mismo_autor,retrieval_q3_mismo_genero,retrieval_q4_popularidad_pais,retrieval_q5_trending,retrieval_hibrido,retrieval_q6_productos_asociados

def evaluar_experimentos(df_test, df_train_features, modelo_lgbm_reg=None,modelo_lgbm_ranker=None):
    
    # 1. Definimos el diccionario de Retrievals a evaluar
    estrategias_retrieval = {
        "Q1_Populares": lambda u: retrieval_q1_populares(df_train_features, u, top_n=100),
        "Q2_Mismo_Autor" : lambda u: retrieval_q2_mismo_autor(df_train_features, u, top_n=100),
        "Q3_Mismo_Genero" : lambda u: retrieval_q3_mismo_genero(df_train_features, u, top_n=100),
        "Q4_Popularidad_Pais": lambda u: retrieval_q4_popularidad_pais(df_train_features, u, top_n=100),
        "Q5_Trending": lambda u: retrieval_q5_trending(df_train_features, u, top_n=100),
        "Hibrido": lambda u: retrieval_hibrido(df_train_features, u, top_n_por_query=25),
        "Q6_Productos_asociados" : lambda u: retrieval_q6_productos_asociados(df_train_features, u, top_n=20)
    }

    # 2. Definimos los Sistemas de ordenamiento
    estrategias_ordenamiento = ["Ordenamiento_Popularidad"]
    if modelo_lgbm_reg is not None:
        estrategias_ordenamiento.append("Ordenamiento_LGBMRegressor")
        features_del_modelo_regressor = modelo_lgbm_reg.booster_.feature_name()
        #os.makedirs(OUTPUTS_MODELO_TRAIN + 'lgbm_reg/', exist_ok=True) # ELIMINAR ESTA FILA

    if modelo_lgbm_ranker is not None:
        estrategias_ordenamiento.append("Ordenamiento_LGBMRanker")
        features_del_modelo_ranker = modelo_lgbm_ranker.booster_.feature_name()
        #os.makedirs(OUTPUTS_MODELO_TRAIN + 'lgbm_ranker/', exist_ok=True) # ELIMINAR ESTA FILA
    
    pop_dict = df_train_features.groupby("id_libro")["rating_promedio_te_por_libro"].max().to_dict()
    media_global = df_train_features["rating"].mean()
    
    resultados_totales = []

    
    # Iteramos sobre la matriz de experimentos (Retrieval x Ordenamiento)
    for nombre_retrieval, func_retrieval in estrategias_retrieval.items():
        if nombre_retrieval != 'Q6_Productos_asociados':
            for nombre_ordenamiento in estrategias_ordenamiento:
                
                print(f"\n ***************** Evaluando: Retrieval=[{nombre_retrieval}] | Ordenamiento=[{nombre_ordenamiento}]***************")
                inicio = time.time()

                checkpoint_file = OUTPUTS_CHECKPOINTS+ f"ckpt_{nombre_retrieval}_{nombre_ordenamiento}.csv"
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
                    
                    if id_lector in usuarios_procesados:
                        continue
                        
                    # --- FASE 1: RETRIEVAL ---
                    libros_candidatos = func_retrieval(id_lector)
                    
                    
                    subset_test = df_test[df_test["id_lector"] == id_lector]
                    true_relevance_dict = subset_test.set_index("id_libro")["rating"].to_dict()
                    libros_test_reales = set(true_relevance_dict.keys())
                    
                    hits = len(set(libros_candidatos).intersection(libros_test_reales))
                    recall_retrieval = hits / len(libros_test_reales) if len(libros_test_reales) > 0 else 0

                    # --- FASE 2: ORDENAMIENTO ---
                    predicted_scores_dict = {}

                    
                    if len(libros_candidatos) > 0:
                        if nombre_ordenamiento == "Ordenamiento_Popularidad":
                            for lib in libros_candidatos:
                                predicted_scores_dict[lib] = pop_dict.get(lib, media_global)
                                
                        elif nombre_ordenamiento == "Ordenamiento_LGBMRegressor" and modelo_lgbm_reg is not None:
                            df_candidatos = pd.DataFrame({
                                    "id_lector": [id_lector] * len(libros_candidatos),
                                    "id_libro": libros_candidatos
                                })
                            
                            df_candidatos_feat = pipeline_feature_engineering(df_candidatos, state="test", df_historia=df_train_features)
                            
                            X_candidatos = df_candidatos_feat[features_del_modelo_regressor].copy()

                            cat_features_eval = [c for c in CAT_FEATURES if c in features_del_modelo_regressor]
                            for c in cat_features_eval:
                                # Alineamos categorías con el train para que LightGBM no explote
                                X_candidatos[c] = pd.Categorical(X_candidatos[c], categories=df_train_features[c].cat.categories)


                            y_pred = modelo_lgbm_reg.predict(X_candidatos)
                            predicted_scores_dict = dict(zip(libros_candidatos, y_pred))

                        elif nombre_ordenamiento == "Ordenamiento_LGBMRanker" and modelo_lgbm_ranker is not None:
                            df_candidatos = pd.DataFrame({
                                    "id_lector": [id_lector] * len(libros_candidatos),
                                    "id_libro": libros_candidatos
                                })
                            
                            df_candidatos_feat = pipeline_feature_engineering(df_candidatos, state="test", df_historia=df_train_features)
                            
                            X_candidatos = df_candidatos_feat[features_del_modelo_ranker].copy()

                            cat_features_eval = [c for c in CAT_FEATURES if c in features_del_modelo_ranker]
                            for c in cat_features_eval:
                                # Alineamos categorías con el train para que LightGBM no explote
                                X_candidatos[c] = pd.Categorical(X_candidatos[c], categories=df_train_features[c].cat.categories)


                            y_pred = modelo_lgbm_ranker.predict(X_candidatos)
                            predicted_scores_dict = dict(zip(libros_candidatos, y_pred))
                    # --- FASE 3: MÉTRICAS DE ORDENAMIENTO (NDCG) ---
                    # ¡EL PARCHE PARA NO INFLAR EL NDCG EN COLD-START!
                    if len(libros_candidatos) == 0:
                        ndcg_val = 0.0
                    else:
                        #id_libros_totales = list(set(list(true_relevance_dict.keys()) + list(predicted_scores_dict.keys())))
                        # 1. ORDENAMOS ALFABÉTICAMENTE para destruir el sesgo de la concatenación
                        id_libros_totales = sorted(list(set(list(true_relevance_dict.keys()) + list(predicted_scores_dict.keys()))))

                        if len(id_libros_totales) > 0:
                            y_true = np.asarray([[true_relevance_dict.get(id_libro, 0) for id_libro in id_libros_totales]])
                            #y_score = np.asarray([[predicted_scores_dict.get(id_libro, 0) for id_libro in id_libros_totales]])
                            
                            # El -99999 manda a los libros no recuperados al fondo
                            y_score_base = np.asarray([[predicted_scores_dict.get(id_libro, -99999) for id_libro in id_libros_totales]])

                            # 2. ROMPEMOS EMPATES: Sumamos un ruido aleatorio invisible (ej: 0.000004)
                            y_score = y_score_base + np.random.uniform(0, 1e-5, size=y_score_base.shape)

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
                        print(f"Tamaño libros candidatos : {len(libros_candidatos)}")

                # --- FIN DEL BUCLE DE USUARIOS (Cálculo promedio del experimento) ---
                resultados_totales.append({
                    "Retrieval": nombre_retrieval,
                    "Ordenamiento_system": nombre_ordenamiento,
                    "Recall_Retrieval_Promedio": np.mean(recall_retrieval_lista),
                    "NDCG@20_Promedio": np.mean(ndcg_lista),
                    "Tiempo_Segundos": round(time.time() - inicio, 2)
                })
                
                print(f" -> NDCG@20: {np.mean(ndcg_lista):.4f} | Recall: {np.mean(recall_retrieval_lista):.4f}")
        elif  nombre_retrieval == 'Q6_Productos_asociados':
            inicio = time.time()
            checkpoint_file = OUTPUTS_CHECKPOINTS+ f"ckpt_{nombre_retrieval}.csv"
            usuarios_procesados = {}
            if os.path.exists(checkpoint_file):
                print(f" -> Encontrado checkpoint previo: {checkpoint_file}. Retomando...")
                df_ckpt = pd.read_csv(checkpoint_file)
                for _, row in df_ckpt.iterrows():
                    usuarios_procesados[row["id_lector"]] = {
                        "recall": row["recall"],
                        "ndcg": row["ndcg"]
                    }
            ndcg_lista = [v["ndcg"] for v in usuarios_procesados.values()]
            recall_retrieval_lista = [v["recall"] for v in usuarios_procesados.values()]

            usuarios_test = df_test["id_lector"].unique() 
            nuevos_procesados_en_esta_sesion = 0

            for id_lector in usuarios_test:
                if id_lector in usuarios_procesados:
                    continue
                libros_candidatos_df = func_retrieval(id_lector)
                predicted_scores_dict = libros_candidatos_df.set_index('id_recomendado')['score_final'].to_dict()
                libros_candidatos = list(predicted_scores_dict.keys())
                subset_test = df_test[df_test["id_lector"] == id_lector]
                true_relevance_dict = subset_test.set_index("id_libro")["rating"].to_dict()
                libros_test_reales = set(true_relevance_dict.keys())
                
                hits = len(set(libros_candidatos).intersection(libros_test_reales))
                recall_retrieval = hits / len(libros_test_reales) if len(libros_test_reales) > 0 else 0

                
                if len(libros_candidatos) == 0 :
                    ndcg_val = 0.0
                else:
                    id_libros_totales = sorted(list(set(libros_candidatos +list(true_relevance_dict.keys()))))
                    if len(id_libros_totales) > 0 :
                        y_true =np.asarray( [[true_relevance_dict.get(id_libro,0) for id_libro in id_libros_totales]])
                        y_score_base = np.asarray([[predicted_scores_dict.get(id_libro,-99999) for id_libro in id_libros_totales]])
                        y_score = y_score_base + np.random.uniform(0,1e-5 ,size =y_score_base.shape)
                        ndcg_val = ndcg_score(y_true , y_score , k=20)
                    else :
                        ndcg_val = 0.0
                
                ndcg_lista.append(ndcg_val)
                recall_retrieval_lista.append(recall_retrieval)
                registro_nuevo = pd.DataFrame([{
                    "id_lector" :id_lector,
                    "recall": recall_retrieval,
                    "ndcg" : ndcg_val
                }])
                
                registro_nuevo.to_csv(checkpoint_file, mode='a', header=not os.path.exists(checkpoint_file), index=False)

                nuevos_procesados_en_esta_sesion += 1
                if nuevos_procesados_en_esta_sesion % 50 == 0:
                    print(f"   -> Procesados {nuevos_procesados_en_esta_sesion} usuarios...")
                    print(f"Tamaño libros candidatos : {len(libros_candidatos)}")
                
            resultados_totales.append({
                "Retrieval": nombre_retrieval,
                "Ordenamiento_system": 'indice_jaccard',
                "Recall_Retrieval_Promedio": np.mean(recall_retrieval_lista),
                "NDCG@20_Promedio": np.mean(ndcg_lista),
                "Tiempo_Segundos": round(time.time() - inicio, 2)
            })
            print(f" -> NDCG@20: {np.mean(ndcg_lista):.4f} | Recall: {np.mean(recall_retrieval_lista):.4f}")
            
            

        
    
    # Al finalizar todos los cruces, exporta la tabla bonita
    df_resultados = pd.DataFrame(resultados_totales)
    df_resultados.to_csv(OUTPUTS_RESULTADOS +"resultados_experimentos.csv", index=False)
    print(f"Resultados guardados con éxito en {OUTPUTS_RESULTADOS} resultados_experimentos.csv")
    
    return df_resultados

