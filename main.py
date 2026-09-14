import os
import pandas as pd
import numpy as np

# Importaciones de tus módulos locales
from src.config import OUTPUTS_DATASETS, TARGET, FEATURES, FEATURES_OUT
from src.utils import folders_creation, train_test_split_dfs,model_baseline_train_lgbm_reg ,model_baseline_train_lgbm_ranker, calcular_feature_importance,evaluar_experimentos      
from src.feat_eng import pipeline_feature_engineering
from src.retrievals import (                                # Asegurate de importar tus retrievals si los usas acá
    retrieval_q1_populares, 
    retrieval_q2_mismo_autor, 
    retrieval_q3_mismo_genero, 
    retrieval_q4_popularidad_pais, 
    retrieval_q5_trending, 
    retrieval_hibrido
)


def main():
    print("=" * 60)
    print("INICIANDO PIPELINE DE SISTEMA DE RECOMENDACIÓN DE LIBROS")
    print("=" * 60)

    # 1. Asegurar que existan los directorios de outputs necesarios
    folders_creation()
    #os.makedirs("outputs/modelo", exist_ok=True)

# 2. Split de Train y Test (Basado en los últimos 20 libros por usuario)
    print("\n--- PASO 1: Generando / Cargando splits de Train y Test ---")
    df_train, df_test = train_test_split_dfs()
    print(f"Filas en Train: {len(df_train)} | Filas en Test: {len(df_test)}")

    # 3. Feature Engineering sobre Train
    # Esto calcula métricas históricas, guarda el archivo en disco y devuelve el df enriquecido
    print("\n--- PASO 2: Ejecutando Feature Engineering en Train ---")
    df_train_features = pipeline_feature_engineering(df_train, state="entrenamiento")

    # Limpieza de seguridad por si quedaron nulos en el target
    df_train_features = df_train_features.dropna(subset=[TARGET])

    # 4. Definición de Features para el Modelo
    # Excluimos de las features de entrenamiento aquellas que son IDs, textos libres o el target
    features_a_usar = [f for f in FEATURES if f not in FEATURES_OUT]
    print(f"\nFeatures seleccionadas para entrenar ({len(features_a_usar)}):")
    print(features_a_usar)

    # 5. Entrenamiento del Modelo Base (LightGBM Regressor)
    print("\n--- PASO 3: Entrenando modelo LightGBM ---")
    modelo_lgbm_reg = model_baseline_train_lgbm_reg(
        df_train=df_train_features, 
            features_seleccionadas=features_a_usar, 
            seed=42)

    modelo_lgbm_ranker = model_baseline_train_lgbm_ranker( 
        df_train=df_train_features, 
        features_seleccionadas=features_a_usar, 
        seed=42
    )

    # Calculo FEAT IMP
    df_importances_reg= calcular_feature_importance(modelo_lgbm_reg,'lgbm_regressor')
    df_importances_ranker= calcular_feature_importance(modelo_lgbm_ranker,'lgbm_ranker')

    # 6. Evaluación de Experimentos (Retrievals x Rankers)
    print("\n--- PASO 4: Evaluando matriz de experimentos (Retrievals + Ranking) ---")

    df_resultados = evaluar_experimentos(
        df_test=df_test, 
        df_train_features=df_train_features, 
        modelo_lgbm_reg=modelo_lgbm_reg ,
        modelo_lgbm_ranker=modelo_lgbm_ranker
    )

    print("\n" + "=" * 60)
    print("RESULTADOS FINALES DE LOS EXPERIMENTOS:")
    print("=" * 60)
    print(df_resultados.to_string(index=False))

    print('*'*30 +"FIN DEL PROCESO" + '*'*30 )



if __name__ == "__main__":
    main()