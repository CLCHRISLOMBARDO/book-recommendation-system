# DATASETS FILES
BASE_DB = "dataset/data.db"
TABLA_MODELO_FULL = "interacciones"
TABLA_LIBROS= "libros"
TABLA_LECTORES = "lectores"
EJEMPLO_FILE = "dataset/ejemplo.csv"

# OUTPUTS
       # MODELO FINAL
OUTPUTS_MODELO_FINAL = "outputs/scripts/modelos_finales/"

OUTPUTS_DATASETS = "outputs/datasets/"

# FEATURES
FEATURES_OUT = [
    'id_lector', 
    'id_libro', 
    'nombre',                  # Ojo: en tu SQL cruzado quedó como 'nombre' (no 'nombre_lector')
    'resumen',                 # Texto libre, LightGBM no lo entiende sin NLP
    'titulo',                  # Texto libre
    'fecha_interaccion',       # Fecha cruda
    'vive_en',                 # Lo pasamos acá porque ahora usamos 'pais'
    'rating_promedio_crudo_por_lector' # Lo dejamos afuera a favor de la variable TE (Target Encoded)
]

FEATURES = [
    # --- Features Demográficas / Estáticas (Lector) ---
    'genero',                  # Género del lector
    'nacimiento',              # Año de nacimiento
    'pais',                    # La nueva feature limpia que creamos
    
    # --- Features de Contenido / Estáticas (Libro) ---
    'autor',
    'genero_libro',
    'editorial',
    'anio_edicion',
    
    # --- Features de Comportamiento (Lector) ---
    'volumen_lectura_por_lector',
    'desviacion_estandar_rating_por_lector',
    'genero_favorito_por_lector',
    'ratio_libros_antiguos_vs_nuevos_por_lector',
    'rating_promedio_te_por_lector',
    'pct_autor_favorito_por_lector',
    'pct_genero_favorito_por_lector',
    
    # --- Features de Comportamiento / Popularidad (Libro) ---
    'popularidad_count_por_libro',
    'popularidad_rating_mean_por_libro',
    'rating_promedio_te_por_libro',
    
    # --- Features de Interacción (Candidato vs Historial del Lector) ---
    'pct_lecturas_autor_candidato_por_par_libroylector',
    'pct_lecturas_genero_candidato_por_par_libroylector'
]

CAT_FEATURES = [
    'genero',                     # Género del lector
    'pais',                       # País limpio
    'autor',                      # Autor del libro
    'genero_libro',               # Género del libro
    'editorial',                  # Editorial
    'genero_favorito_por_lector'  # El género más leído por ese usuario
]

#FEATURES_LECTOR=['genero', 'vive_en', 'nacimiento']
#FEATURES_LIBRO=['autor','genero_libro','editorial', 'anio_edicion']
#FEATURES_COMPORTAMIENTO=['volumen_lectura', 'rating_promedio_usuario_te',
 #                        'desviacion_estandar_rating', 'genero_favorito_usuario',
  #                       'cant_antiguos', 'cant_nuevos', 'ratio_libros_antiguos_vs_nuevos']

TARGET='rating'

# SPLIT TRAIN TEST
SEED=42
seeds=[24,50,15,48,59,63,20,15,45,75,85,62,100]

