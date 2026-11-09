query_caso_1 =  """
    WITH caract_lector as (
        select distinct id_lector, genero, vive_en, nacimiento,
        volumen_lectura, rating_promedio_usuario_te, desviacion_estandar_rating, genero_favorito_usuario,
        cant_antiguos, cant_nuevos, ratio_libros_antiguos_vs_nuevos
        from BASE_MODELO_FULL_FEAT_ENG 
        where id_lector = ?
    ),
    libros_leidos as (	
        select id_libro, autor, genero_libro, editorial, anio_edicion 
        from BASE_MODELO_FULL_FEAT_ENG 
        where id_lector = ?
    ),
    libros_a_predecir as (
        select id_libro, ? as id_lector, autor, genero as genero_libro, editorial, CAST(anio_edicion AS INTEGER) AS anio_edicion
        from libros 
        where id_libro not in (select id_libro from libros_leidos) 
        and genero in (select genero_libro from libros_leidos) 
    ),
    base_a_predecir as (
        select *
        from libros_a_predecir lp
        inner join caract_lector cl 
        using(id_lector)
    )
    select * from base_a_predecir
    """

query_caso_2 =  """
                        WITH 
global_stats AS (
    SELECT AVG(rating * 1.0) AS rating_global
    FROM BASE_MODELO_FULL
),
metricas_lector AS (
    SELECT 
        i.id_lector,
        COUNT(i.id_libro) AS volumen_lectura,
        AVG(i.rating * 1.0) AS rating_promedio_crudo,
        
        -- Desviación estándar matemática calculada a mano
        CASE 
            WHEN COUNT(i.id_libro) > 1 THEN 
                SQRT(MAX(0.0, AVG(i.rating * i.rating * 1.0) - (AVG(i.rating * 1.0) * AVG(i.rating * 1.0))))
            ELSE 0.0 
        END AS desviacion_estandar_rating,
        
        -- Conteo de libros antiguos vs modernos (criterio año 2000)
        SUM(CASE WHEN CAST(l.anio_edicion as INTEGER) < 2000 THEN 1 ELSE 0 END) AS cant_antiguos,
        SUM(CASE WHEN CAST(l.anio_edicion as INTEGER) >= 2000 THEN 1 ELSE 0 END) AS cant_nuevos
        -- (ERROR CORREGIDO: se sacó la coma de arriba)
    FROM interacciones i 
    INNER JOIN libros l USING(id_libro)
    WHERE i.id_lector = ?
),
generos_contados AS (
    SELECT 
        i.id_lector,
        l.genero AS genero_favorito_usuario,
        COUNT(*) AS cant,
        ROW_NUMBER() OVER (
            PARTITION BY i.id_lector 
            ORDER BY COUNT(*) DESC
        ) AS ranking_genero
    FROM interacciones i 
    INNER JOIN libros l USING(id_libro)
    WHERE l.genero IS NOT NULL AND l.genero != ''
    AND i.id_lector = ?
    GROUP BY i.id_lector, l.genero
),

genero_preferido AS (
    SELECT id_lector, genero_favorito_usuario
    FROM generos_contados
    WHERE ranking_genero = 1
),

STATS_POR_LECTOR AS (
    SELECT 
        m.id_lector,
        m.volumen_lectura,
        m.rating_promedio_crudo,
        
        ROUND(((m.volumen_lectura * m.rating_promedio_crudo) + (10.0 * g.rating_global)) / (m.volumen_lectura + 10.0), 4) AS rating_promedio_usuario_te,
        ROUND(m.desviacion_estandar_rating, 4) AS desviacion_estandar_rating,
        COALESCE(gp.genero_favorito_usuario, 'DESCONOCIDO') AS genero_favorito_usuario,
        m.cant_antiguos,
        m.cant_nuevos,
        ROUND((m.cant_antiguos * 1.0) / (m.cant_nuevos + 1.0), 4) AS ratio_libros_antiguos_vs_nuevos
    FROM metricas_lector m
    CROSS JOIN global_stats g
    LEFT JOIN genero_preferido gp ON gp.id_lector = m.id_lector
),

CRUCE_FINAL AS (
    SELECT 
        i.id_lector, i.id_libro, i.rating,
        l.autor, l.genero AS genero_libro, l.editorial, CAST(l.anio_edicion AS INTEGER) AS anio_edicion,
        s.volumen_lectura, s.rating_promedio_crudo, s.rating_promedio_usuario_te,
        s.desviacion_estandar_rating, s.genero_favorito_usuario, s.cant_antiguos, s.cant_nuevos, s.ratio_libros_antiguos_vs_nuevos
    FROM interacciones i
    INNER JOIN libros l USING(id_libro)
    LEFT JOIN STATS_POR_LECTOR s ON s.id_lector = i.id_lector
    WHERE i.id_lector = ?
), -- (ERROR CORREGIDO: el WHERE va adentro y se agregó la coma)

caract_lector AS (
    SELECT DISTINCT id_lector, NULL AS genero, NULL AS vive_en, CAST( NULL AS INTEGER) AS nacimiento,
    volumen_lectura, rating_promedio_usuario_te, desviacion_estandar_rating, genero_favorito_usuario,
    cant_antiguos, cant_nuevos, ratio_libros_antiguos_vs_nuevos
    FROM CRUCE_FINAL 
    WHERE id_lector = ?
),
libros_leidos AS (	
    SELECT id_libro, autor, genero_libro, editorial, anio_edicion 
    FROM CRUCE_FINAL 
    WHERE id_lector = ?
),
libros_a_predecir AS (
    SELECT id_libro, ? AS id_lector, autor, genero AS genero_libro, editorial, CAST(anio_edicion AS INTEGER) AS anio_edicion
    FROM libros 
    WHERE id_libro NOT IN (SELECT id_libro FROM libros_leidos) 
    AND genero IN (SELECT genero_libro FROM libros_leidos) 
),
base_a_predecir AS (
    SELECT *
    FROM libros_a_predecir lp
    INNER JOIN caract_lector cl USING(id_lector)
)
SELECT * FROM base_a_predecir
                """
query_caso_3 =  """
                    WITH caract_lector AS (
		SELECT 
        DISTINCT id_lector, 
        genero, 
        vive_en, 
        CAST(nacimiento AS INTEGER) AS nacimiento,
        CAST(NULL AS REAL) AS volumen_lectura, 
        CAST(NULL AS REAL) AS rating_promedio_usuario_te, 
        CAST(NULL AS REAL) AS desviacion_estandar_rating, 
        NULL AS genero_favorito_usuario, -- Este queda NULL crudo porque es categórico/texto
        CAST(NULL AS REAL) AS cant_antiguos, 
        CAST(NULL AS REAL) AS cant_nuevos, 
        CAST(NULL AS REAL) AS ratio_libros_antiguos_vs_nuevos
    FROM lectores 
    WHERE id_lector = ?
),

-- Buscamos los 500 libros más populares entre los lectores de su MISMO año de nacimiento
libros_candidatos AS (
    SELECT 
        id_libro, 
        autor, 
        genero_libro, 
        editorial, 
        anio_edicion,
        COUNT(id_libro) AS cantidad_interacciones
    FROM BASE_MODELO_FULL_FEAT_ENG 
    WHERE nacimiento = (SELECT nacimiento FROM caract_lector)
    GROUP BY id_libro, autor, genero_libro, editorial, anio_edicion
    ORDER BY cantidad_interacciones DESC
    LIMIT 500 
),

base_a_predecir AS (
    SELECT 
        lc.id_libro, 
        cl.id_lector, 
        lc.autor, 
        lc.genero_libro, 
        lc.editorial, 
        lc.anio_edicion,
        cl.genero, 
        cl.vive_en, 
        cl.nacimiento, 
        cl.volumen_lectura, 
        cl.rating_promedio_usuario_te, 
        cl.desviacion_estandar_rating, 
        cl.genero_favorito_usuario, 
        cl.cant_antiguos, 
        cl.cant_nuevos, 
        cl.ratio_libros_antiguos_vs_nuevos
    FROM libros_candidatos lc
    CROSS JOIN caract_lector cl
)

SELECT * FROM base_a_predecir

                """
