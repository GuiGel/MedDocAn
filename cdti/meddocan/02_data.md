# Clinical Corpus

## [Anotación](https://github.com/PlanTL-GOB-ES/SPACCC_MEDDOCAN#annotation-tool)

 El corpus MEDDOCAN es un corpus sintético de casos clínicos enriquecido con expresiones PHI (Información Sanitaria Protegida). El corpus MEDDOCAN, de 1.000 casos clínicos, fue seleccionado manualmente por un médico en ejercicio y enriquecido con frases de PHI por documentalistas sanitarios, añadiendo información de PHI procedente de resúmenes de alta y de historias clínicas de genética médica. 
 Para llevar a cabo la anotación manual, el equipo construyó las primeras pautas públicas de PHI en español [16], siguiendo las especificaciones derivadas del Reglamento General de Protección de Datos (GDPR) de la UE, así como las pautas y tipos de anotación definidos por las vías de desidentificación de i2b2, basadas en la Ley de Portabilidad y Responsabilidad del Seguro Médico (HIPAA) de Estados Unidos. La elaboración de estas directrices de anotación supuso una retroalimentación activa a lo largo de seis meses por parte de un equipo híbrido de nueve personas con experiencia tanto en sanidad como en PNL, lo que dio como resultado un documento de 28 páginas [^1] que se ha distribuido junto con el corpus. Junto con las reglas de anotación, se proporcionaron ejemplos ilustrativos para facilitar al máximo la interpretación y el uso de las directrices. El corpus de MEDDOCAN se muestreó aleatoriamente en tres subconjuntos: el conjunto de entrenamiento, que contenía 500 casos clínicos, y los conjuntos de desarrollo y de prueba de 250 casos clínicos cada uno. Estos casos clínicos se anotaron manualmente utilizando una versión personalizada de AnnotateIt [^2]. A continuación, se utilizó el kit de herramientas de anotación BRAT para corregir los errores y añadir las anotaciones que faltaban, logrando un acuerdo entre anotadores (IAA) del 98% (calculado con 50 documentos). Junto con el conjunto de pruebas, publicaron una colección adicional de 3.501 documentos (conjunto de fondo[^3] ) para asegurarse de que los equipos participantes no pudieran hacer correcciones manuales y también para promover que estos sistemas fueran potencialmente capaces de escalar a colecciones de datos más grandes. Las directrices de anotación de MEDDOCAN definieron un total de 29 tipos de entidades. {numref}`true-entity-distribution` resume la lista de tipos de entidad sensibles definidos para la pista de MEDDOCAN y el número de ocurrencias entre los conjuntos de entrenamiento, desarrollo y prueba.


```{table} Distribución del tipo de entidad entre los juegos de datos
:name: true-entity-distribution
|               Tipo               | Train | Dev | Test | Total |
| :------------------------------: | :---: | :-: | :--: | :---: |
|            TERRITORIO            | 1875  | 987 | 956  | 3818  |
|              FECHAS              | 1231  | 724 | 611  | 2566  |
|      EDAD SUJETO ASISTENCIA      | 1035  | 521 | 518  | 2074  |
|     NOMBRE SUJETO ASISTENCIA     | 1009  | 503 | 502  | 2014  |
|    NOMBRE PERSONAL SANITARIO     | 1000  | 497 | 501  | 1998  |
|      SEXO SUJETO ASISTENCIA      |  925  | 455 | 461  | 1841  |
|              CALLE               |  862  | 434 | 413  | 1709  |
|               PAIS               |  713  | 347 | 363  | 1423  |
|       ID SUJETO ASISTENCIA       |  567  | 292 | 283  | 1142  |
|        CORREO ELECTRONICO        |  469  | 241 | 249  |  959  |
| ID TITULACION PERSONAL SANITARIO |  471  | 226 | 234  |  931  |
|         ID ASEGURAMIENTO         |  391  | 194 | 198  |  783  |
|             HOSPITAL             |  255  | 140 | 130  |  525  |
|   FAMILIARES SUJETO ASISTENCIA   |  243  | 92  |  81  |  416  |
|           INSTITUCION            |  98   | 72  |  67  |  237  |
|     ID CONTACTO ASISTENCIAL      |  77   | 32  |  39  |  148  |
|         NUMERO TELEFONO          |  58   | 25  |  26  |  109  |
|            PROFESION             |  24   |  4  |  9   |  37   |
|            NUMERO FAX            |  15   |  6  |  7   |  28   |
|     OTROS SUJETO ASISTENCIA      |   9   |  6  |  7   |  22   |
|           CENTRO SALUD           |   6   |  2  |  6   |  14   |
|   ID EMPLEO PERSONAL SANITARIO   |   0   |  1  |  0   |   1   |
| IDENTIF VEHICULOS NRSERIE PLACAS |   0   |  0  |  0   |   0   |
|   IDENTIF DISPOSITIVOS NRSERIE   |   0   |  0  |  0   |   0   |
|     NUMERO BENEF PLAN SALUD      |   0   |  0  |  0   |   0   |
|             URL WEB              |   0   |  0  |  0   |   0   |
|       DIREC PROT INTERNET        |   0   |  0  |  0   |   0   |
|        IDENTF BIOMETRICOS        |   0   |  0  |  0   |   0   |
|       OTRO NUMERO IDENTIF        |   0   |  0  |  0   |   0   |
```

El corpus de MEDDOCAN se distribuyó en texto plano en codificación UTF-8, donde cada caso clínico se almacenó como un único archivo, mientras que las anotaciones de PHI se publicaron en el formato BRAT, lo que hace que la visualización de los resultados sea sencilla, como se puede ver en la {numref}`Figura {number} <brat annotator visualization>`. Para este tema, también facilitaron un script de conversión[^4] entre el formato de anotación BRAT y el formato de anotación utilizado por el esfuerzo anterior de i2b2, para facilitar la comparación y adaptación de los sistemas anteriores utilizados para los textos en inglés.

```{figure} ../figures/meddocan-brat-visualization.png
:align: center
:name: brat annotator visualization

Un ejemplo de anotación de MEDDOCAN visualizada mediante la interfaz de anotación BRAT
```

[^1]: https://github.com/PlanTL-GOB-ES/SPACCC_MEDDOCAN/blob/master/guidelines/gu%C3%ADas-de-anotaci%C3%B3n-de-informaci%C3%B3n-de-salud-protegida.pdf
[^2]: https://annotateit.org/
[^3]: El conjunto de datos de referencia incluía los conjuntos de entrenamiento, desarrollo y prueba, y una colección adicional de 2.751 casos clínicos (en total, 3.751 casos clínicos). 
[^4]: https://github.com/PlanTL-SANIDAD/MEDDOCAN-Format-Converter-Script

## Preparación de los datos

Tras retomar la descripción del corpus por parte de sus autores, veamos con más detalle en qué consiste la preparación de los datos. El corpus MEDDOCAN consiste en casos clínicos escritos en español y enriquecidos manualmente con expresiones PHI. Se considera un número total de {glue:}`phi_class_num` categorías PHI que muestran una alta variabilidad de frecuencia [^5].
El número calculado de categorías PHI se puede encontrar en la {numref}`Figura {number}<PHI class statistics>`.

```{glue:figure} phi_statistics
:figwidth: 500px
:align: center
:name: "PHI class statistics"

Las categorías de PHI calculadas a partir de los conjuntos de datos
```

El preprocesamiento y el formateo aplicados al corpus consistieron en los siguientes pasos (véase {numref}`meddocan-pipeline`):

**1. División en párrafos**
: Los documentos se dividieron en párrafos utilizando los saltos de línea de los textos originales. Decidimos trabajar con párrafos en lugar de frases porque las frases reales son difíciles de detectar.

**2. Tokenization**
: Cada párrafo fue tokenizado utilizando un tokenizador personalizado creado con la biblioteca ``spaCy`` [^6] y algunas reglas de tokenización personalizadas adicionales, principalmente para dividir los símbolos de puntuación si no están dentro de una URL, una dirección de correo electrónico o una fecha. Para dividir ciertas palabras con el fin de tener en cuenta los errores de espaciado en el texto original. Por ejemplo, (p. ej. {glue:text}`spacing_error_ex`).

**3. Formato de las etiquetas**
: Las anotaciones con formato Brat de los juegos de datos de entrenamiento y desarrollo se convirtieron en etiquetas a nivel de token siguiendo el esquema **BIO** (Beginning, Inner, Outside). Combinando este esquema de etiquetas con las 22 clases granulares originales de PHI (por ejemplo, para la clase granular **FECHA** tendríamos las etiquetas **B-FECHA**, ****I-FECHA**, más la clase genérica **O**) se obtiene un conjunto final de etiquetas de 45 posibles etiquetas únicas.

Estos 3 pasos se ilustran en {numref}`Figure {number} <Preparación de los datos>`.

```{glue:figure} data_preparation
:figwidth: 800px
:name: "Preparación de los datos"

Una ilustración de los pasos de preparación de datos
```

Las estadísticas finales, incluyendo el número de documentos, párrafos, tokens, tamaño del vocabulario y entidades PHI para cada uno de los conjuntos de datos del corpus preprocesado, pueden consultarse en la {numref}`Figura {number}<doc statistics>`.

```{glue:figure} doc_statistics
:figwidth: 800px
:name: "doc statistics"

Estadísticas finales de los conjuntos de datos pre-procesados
```

[^5]: El esquema de anotación de MEDDOCAN define 29 tipos de entidad PHI como se muestra en {numref}`true-entity-distribution`, pero sólo 22 de ellos aparecen realmente en los conjuntos anotados.
[^6]: https://spacy.io/