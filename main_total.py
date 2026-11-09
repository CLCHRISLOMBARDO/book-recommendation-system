import os
import pandas as pd
import numpy as np

# Importaciones de tus módulos locales
from src.config import OUTPUTS_DATASETS, TARGET, FEATURES, FEATURES_OUT
from src.utils import folders_creation, cracion_df_total,model_baseline_train_lgbm_reg ,model_baseline_train_lgbm_ranker, calcular_feature_importance,evaluar_experimentos      
from src.feat_eng import pipeline_feature_engineering
from src.filtrado_colaborativo import filtrado_colaborativo
from src.construccion_cache import construir_cache
from src.entrenar_ease import obtener_modelo_ease
import src.retrievals as ret
import src.rankers as rnk


def main():
    print("=" * 60)
    print("INICIANDO PIPELINE DE SISTEMA DE RECOMENDACIÓN DE LIBROS")
    print("=" * 60)

    #if state =='entrenamiento': 
    print('*'*30+f"INICIANDO CON STATE : {'entrenamiento'}")
    # 1. Asegurar que existan los directorios de outputs necesarios
   

    # 2. Split de Train y Test (Basado en los últimos 20 libros por usuario)
    print("\n--- PASO 1: Generando / Cargando splits de Train y Test ---")
    df_total = cracion_df_total()

    
        

    # 3. Feature Engineering sobre Train
    # Esto calcula métricas históricas, guarda el archivo en disco y devuelve el df enriquecido
    print("\n--- PASO 2: Ejecutando Feature Engineering en Train ---")
    df_total_features = pipeline_feature_engineering(df_total, state='entrenamiento_final')


 
    
    # 6. FILTRADO COLABORATIVO
    m_lectores, m_libros = filtrado_colaborativo(state='entrenamiento_final')
    
    # 8. EntreNAS ease
    modelo_ease = obtener_modelo_ease(df_total, state="entrenamiento_final")
    
    # 7. Construccion del caché
    cache = construir_cache(df_total_features ,modelo_lgbm_reg=None, modelo_lgbm_ranker=None,modelo_ease=modelo_ease, state='entrenamiento_final')
 
    
    
    volumen_lectura_por_lector = df_total_features.groupby("id_lector").agg(volumen_lectura_por_lector=("volumen_lectura_por_lector",'count' )).reset_index()
    id_lectores = volumen_lectura_por_lector.id_lector.to_list()
    id_lectores_menor_5_libros = volumen_lectura_por_lector.loc[volumen_lectura_por_lector['volumen_lectura_por_lector']<=5,'id_lector'].to_list()
    id_lectores_mayor_5_libros = volumen_lectura_por_lector.loc[volumen_lectura_por_lector['volumen_lectura_por_lector']>5,'id_lector'].to_list()
    
    
    libros_recomendados_resultados=pd.DataFrame({'id_lector':[], 'id_libro':[],'score':[]})
    for id_lector in id_lectores:
        if id_lector in id_lectores_menor_5_libros:
            candidatos=ret.retrieval_q4_popularidad_pais(id_lector, 100, cache)
            scores_predichos_dict = rnk.ranker_popularidad(id_lector, candidatos, cache=cache)
            scores_predichos_sorted = dict(sorted(scores_predichos_dict.items(),key=lambda item:item[1] , reverse=True))
            if len(scores_predichos_sorted) < 20 :
                faltantes = 20-len(scores_predichos_sorted)
                completado_dict = {f'completado_{i}':-1 for i in range(0,faltantes)}
                scores_predichos_sorted.update(completado_dict)
            
            libros_recomendacion_i_df= pd.DataFrame({'id_lector':[id_lector]*20,
                                                    'id_libro':list(scores_predichos_sorted.keys())[:20] ,
                                                    'score':list(scores_predichos_sorted.values())[:20]})
            
            libros_recomendados_resultados = pd.concat([libros_recomendados_resultados,libros_recomendacion_i_df])
            
            
        elif id_lector in id_lectores_mayor_5_libros:
            candidatos=ret.retrieval_q7_ease(id_lector, 100, cache)
            scores_predichos_dict = rnk.ranker_ease(id_lector, candidatos, cache=cache)
            scores_predichos_sorted = dict(sorted(scores_predichos_dict.items(),key=lambda item:item[1] , reverse=True))
            if len(scores_predichos_sorted) < 20 :
                faltantes = 20-len(scores_predichos_sorted)
                completado_dict = {f'completado_{i}':-1 for i in range(0,faltantes)}
                scores_predichos_sorted.update(completado_dict)
            
            libros_recomendacion_i_df= pd.DataFrame({'id_lector':[id_lector]*20,
                                                    'id_libro':list(scores_predichos_sorted.keys())[:20] ,
                                                    'score':list(scores_predichos_sorted.values())[:20]})
            
            libros_recomendados_resultados = pd.concat([libros_recomendados_resultados,libros_recomendacion_i_df])
    libros_recomendados_resultados[['id_lector','id_libro']].to_csv('resultados_prueba_ret_poppais_ease__rank_pop_ease_top100.csv',index=False)


    print('*'*30 +"FIN DEL PROCESO" + '*'*30 )



if __name__ == "__main__":
    main()