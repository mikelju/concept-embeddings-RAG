# Investigación: espacio conceptual dinámico para RAG y retrieval por expansión semántica

## 0. Resumen ejecutivo

La idea que estamos intentando formalizar puede expresarse así:

> En lugar de representar cada chunk únicamente mediante un embedding denso de dimensionalidad fija, construir para cada corpus un espacio semántico propio en el que cada dimensión corresponda a un concepto interpretable descubierto a partir del propio corpus.

Cada chunk quedaría representado mediante un vector:

    chunk = [peso(concepto_1), peso(concepto_2), ..., peso(concepto_K)]

donde K no tiene que estar fijado de antemano y cada dimensión tiene un significado explícito.

Además, cada concepto tendría su propio embedding semántico convencional:

    concepto_j → embedding_j

Esto permitiría desacoplar dos problemas:

1. **Representar un chunk dentro del espacio conceptual del corpus.**
2. **Traducir la pregunta del usuario al vocabulario conceptual del corpus.**

Después, en vez de limitar el retrieval a los chunks más cercanos a la pregunta, se podría realizar una **expansión iterativa**:

    pregunta
      ↓
    conceptos iniciales
      ↓
    conceptos del corpus semánticamente equivalentes/relacionados
      ↓
    chunks relevantes
      ↓
    conceptos importantes descubiertos en esos chunks
      ↓
    conceptos satélite
      ↓
    nuevos chunks
      ↓
    ...
      ↓
    contexto suficiente

La investigación existente confirma que **cada una de las piezas principales de esta idea ya existe en distintas comunidades**, pero no he encontrado en la literatura revisada una implementación que reúna exactamente todas ellas en una única arquitectura:

- corpus-specific concept space;
- dimensiones conceptuales interpretables y dinámicas;
- embeddings de los conceptos para mapear consultas al vocabulario del corpus;
- representación chunk × concept;
- expansión iterativa guiada por conceptos descubiertos durante el retrieval;
- combinación de esta expansión con estructura documental.

La referencia que más directamente se acerca a la primera parte es **SAE-SPLADE (SIGIR 2026)**, que sustituye el vocabulario de tokens de SPLADE por un espacio latente de conceptos aprendido mediante Sparse Autoencoders. Otras líneas muy próximas son Dynamic Vocabulary / DyVo, learned sparse retrieval, topic modelling semántico, GraphRAG/LightRAG, query expansion y los recientes sistemas de hypergraph RAG. :contentReference[oaicite:0]{index=0}


# 1. Problema que se pretende resolver

## 1.1. Limitación del RAG vectorial convencional

El RAG convencional suele hacer:

    documento
       ↓
    chunking
       ↓
    embedding(chunk)
       ↓
    vector database
       ↓
    similarity(query, chunk)
       ↓
    top-K chunks

Esto funciona muy bien cuando:

    pregunta ≈ contenido que necesito recuperar

Pero falla o se degrada cuando la respuesta requiere **información que no está explícitamente expresada en la pregunta**.

Ejemplo:

    "Prepárame un plan de entrenamiento para una carrera ciclista de 200 km
     teniendo en cuenta mis características."

Para contestar correctamente pueden ser necesarias informaciones sobre:

    entrenamiento
    periodización
    volumen
    intensidad
    ultradistancia
    nutrición
    hidratación
    recuperación
    lesiones
    fatiga
    pacing
    etc.

La pregunta puede contener explícitamente sólo algunas de ellas.

El problema no es simplemente encontrar el chunk más parecido.

El problema es:

> **descubrir qué conocimiento adicional es necesario para construir una respuesta completa y navegar por el corpus hasta encontrarlo.**

---

# 2. Hipótesis de la solución

La hipótesis es que el corpus puede tener su propio "sistema de coordenadas semántico".

Por ejemplo, tras analizar una biblioteca de libros de entrenamiento ciclista podemos descubrir:

    C1  entrenamiento
    C2  periodización
    C3  volumen
    C4  intensidad
    C5  recuperación
    C6  lesiones
    C7  nutrición
    C8  hidratación
    C9  pacing
    C10 potencia
    C11 fatiga
    ...

Cada chunk se representa con respecto a esas dimensiones.

Ejemplo:

               entrenamiento  periodización  volumen  intensidad  nutrición  lesiones

    Chunk 1        0.97           0.91        0.82       0.88       0.05       0.00
    Chunk 2        0.84           0.21        0.10       0.73       0.02       0.02
    Chunk 3        0.12           0.05        0.00       0.00       0.94       0.72
    Chunk 4        0.62           0.31        0.44       0.27       0.71       0.81

Cada fila es un vector.

Por tanto, la matriz:

    CHUNKS × CONCEPTS

es equivalente a una colección de embeddings, pero con una diferencia fundamental:

> **cada dimensión tiene significado semántico explícito.**

---

# 3. Representación matemática conceptual

Si tenemos N chunks y K conceptos:

    X ∈ R^(N × K)

Cada fila:

    X_i = representación conceptual del chunk i

Cada columna:

    X_:j = distribución del concepto j sobre todo el corpus.

Ejemplo:

    X_17 = [0.91, 0.84, 0.17, 0.73, 0.62, ...]

significa:

    chunk 17
        entrenamiento = 0.91
        periodización = 0.84
        volumen       = 0.17
        intensidad    = 0.73
        recuperación  = 0.62
        ...

El valor continuo puede interpretarse como relevancia, activación o confianza, pero esto debe definirse explícitamente en el sistema. No existe un significado universal de "0.73".

---

# 4. Por qué esto se parece a un embedding

Un embedding tradicional:

    texto → [0.12, -0.37, 0.81, ...]

tiene dimensiones cuyo significado no es interpretable directamente.

El embedding conceptual:

    texto →
    [LLM=0.92,
     training=0.81,
     fine-tuning=0.74,
     RAG=0.02,
     ...]

tiene dimensiones semánticamente interpretables.

Por tanto, conceptualmente:

    embedding convencional
        = representación vectorial en espacio latente fijo

    embedding conceptual
        = representación vectorial en espacio semántico inducido por el corpus

Esta segunda representación podría verse como un:

> **Corpus-Induced Semantic Space**

o

> **Dynamic Concept Space**

---

# 5. Qué aportan los valores continuos

Pasar de:

    0 / 1

a:

    0 ... 1

no añade por sí mismo una nueva dimensión.

Mantiene las mismas dimensiones y añade información dentro de cada una.

Por ejemplo:

    training = 1.0

sólo dice presencia fuerte.

Mientras:

    training = 0.82
    training = 0.31
    training = 0.04

permite diferenciar grados de asociación.

Los valores continuos pueden representar:

- intensidad semántica;
- relevancia;
- confianza de la extracción;
- importancia relativa dentro del chunk;
- activación conceptual.

No representan necesariamente relaciones del tipo:

    A --uses--> B

La relación entre conceptos es otra cosa.

---

# 6. El segundo elemento clave: embeddings de los conceptos

Supongamos que el corpus tiene como concepto:

    C17 = "fine-tuning"

La pregunta puede hablar de:

    "adaptar un modelo"

o:

    "ajustar el modelo a una tarea"

Las palabras no coinciden.

Por ello cada concepto del corpus puede tener un embedding semántico:

    embedding("fine-tuning")
    embedding("pretraining")
    embedding("model adaptation")
    embedding("parameter-efficient adaptation")
    ...

Y los conceptos extraídos de la pregunta también:

    embedding("adaptar un modelo")
    embedding("clasificación")
    embedding("entrenamiento")

Entonces podemos hacer:

    concepto_query
        ↓
    embedding
        ↓
    nearest concepts del corpus
        ↓
    columnas de X

Ejemplo:

    "adaptar un modelo"
            ↓
       similitud semántica
            ↓
    fine-tuning       0.94
    model adaptation  0.91
    instruction tuning 0.79
    pretraining       0.42

Así la consulta puede entrar en el espacio conceptual del corpus aunque utilice otro vocabulario.

---

# 7. Los dos espacios no hacen lo mismo

Es importante separar:

## Espacio A: conceptos

Cada concepto tiene un embedding convencional.

Sirve para:

    concepto de la pregunta
        ↓
    concepto del corpus

## Espacio B: chunks × conceptos

Cada chunk tiene un vector cuyas dimensiones son los conceptos del corpus.

Sirve para:

    conceptos relevantes
        ↓
    chunks relevantes

La arquitectura completa sería:

    PREGUNTA
       ↓
    extracción de conceptos
       ↓
    embeddings de esos conceptos
       ↓
    conceptos equivalentes/cercanos del corpus
       ↓
    columnas relevantes de X
       ↓
    chunks relevantes


# 8. El salto importante: retrieval iterativo

Aquí está la parte que diferencia esta propuesta de un simple "tag-based retrieval".

No queremos necesariamente:

    pregunta
       ↓
    top-K chunks
       ↓
    fin

Queremos:

    pregunta
       ↓
    conceptos iniciales
       ↓
    retrieval inicial
       ↓
    analizar los chunks recuperados
       ↓
    descubrir conceptos adicionales
       ↓
    buscar esos conceptos
       ↓
    recuperar nuevos chunks
       ↓
    analizar esos chunks
       ↓
    repetir hasta que la ganancia marginal sea baja


# 9. Ejemplo de expansión en el dominio del ciclismo

Pregunta:

    "Diseña mi preparación para una prueba de 200 km."

Conceptos iniciales:

    entrenamiento
    competición
    200 km
    resistencia
    planificación

Primer retrieval:

    Chunk 12
    Chunk 42
    Chunk 87
    Chunk 105

Analizando esos chunks aparece un patrón:

    entrenamiento
    periodización
    volumen
    intensidad
    recuperación

Pero además aparece consistentemente:

    nutrición
    hidratación
    pacing

El sistema puede interpretar:

    "Estos conceptos son frecuentemente relevantes
     cuando se habla de preparar este tipo de prueba."

Segunda expansión:

    nutrición
    hidratación
    recuperación
    pacing

Segundo retrieval:

    Chunk 211
    Chunk 319
    Chunk 401
    Chunk 428
    ...

Y esos nuevos chunks pueden introducir:

    carbohidratos
    glucógeno
    estrategia energética
    fatiga
    calor
    electrolitos

Tercera expansión:

    carbohidratos
    glucógeno
    electrolitos
    fatiga

El retrieval ya no sigue simplemente la pregunta.

Está **navegando por el espacio conceptual del corpus**.


# 10. Esto es diferente de la intersección de etiquetas

Una estrategia de filtrado sería:

    entrenamiento ∩ competición ∩ nutrición

Pero eso puede ser demasiado restrictivo.

La estrategia propuesta es:

    conceptos iniciales
          ↓
    chunks iniciales
          ↓
    conceptos nuevos relevantes
          ↓
    nuevos chunks
          ↓
    conceptos nuevos
          ↓
    ...

Es decir:

> **intersección local + expansión semántica + iteración**

No queremos necesariamente reducir cada vez más el conjunto.

A veces queremos **ampliarlo inteligentemente**.

---

# 11. Qué significa "relación" en este modelo

No es imprescindible representar:

    A --se_usa_para--> B
    A --es_parte_de--> B
    A --causa--> B

La relación puede ser:

> "Estos conceptos tienen una asociación fuerte dentro del corpus."

o:

> "Estos chunks comparten determinados conceptos."

o:

> "Este concepto aparece sistemáticamente en los contextos en los que aparece aquel otro."

Esto permite construir relaciones de forma **estadística/semántica**, en vez de exigir una ontología explícita de verbos.

---

# 12. Grafo implícito frente a grafo explícito

Una forma útil de verlo:

## Grafo explícito

    A ──uses──> B

La arista tiene un tipo semántico concreto.

## Grafo implícito

    A ───────── B

La fuerza de la conexión viene dada por:

    conceptos compartidos
    similitud conceptual
    coocurrencia
    contexto
    frecuencia
    proximidad dentro de documentos
    relaciones estructurales

En este segundo caso, el grafo puede ser **derivado de la matriz chunk × concept**.

Esto elimina buena parte del problema de decidir qué "verbo" utilizar.

---

# 13. La interpretación como grafo bipartito

La matriz:

    CHUNKS × CONCEPTS

puede interpretarse también como un grafo bipartito:

    CHUNKS                         CONCEPTS

    Chunk 1 ─────────────── LLM
       │
       ├─────────────────── training
       │
       └─────────────────── fine-tuning

    Chunk 2 ─────────────── LLM
       │
       └─────────────────── RAG

    Chunk 3 ─────────────── training
       │
       ├─────────────────── nutrition
       └─────────────────── recovery

Aquí no hemos definido ningún verbo.

La relación se deriva simplemente de la asociación:

    chunk ↔ concepto

A partir de este grafo bipartito se pueden derivar:

    chunk ↔ chunk

y:

    concepto ↔ concepto

mediante sus vecinos compartidos.

Esta observación conecta muy bien con la arquitectura propuesta.


# 14. Conceptos relacionados entre sí

Una vez tienes la matriz X puedes analizar qué conceptos aparecen juntos.

Por ejemplo:

    training ↔ recovery
    training ↔ fatigue
    competition ↔ nutrition
    competition ↔ pacing
    nutrition ↔ hydration
    injury ↔ recovery

Esto genera una especie de:

> **concept association graph**

sin necesidad de etiquetar relaciones con verbos.

La intensidad puede venir dada por la frecuencia o fuerza de coactivación.


# 15. Chunks relacionados entre sí

También puedes comparar filas.

Por ejemplo:

    Chunk A:
        training=.92
        volume=.80
        intensity=.73

    Chunk B:
        training=.89
        volume=.71
        intensity=.69

    Chunk C:
        nutrition=.92
        hydration=.87
        recovery=.73

A y B estarán próximos en el espacio conceptual.

A y C estarán bastante más alejados.

Pero un punto especialmente interesante es que C puede ser recuperado en una **segunda iteración** si A y B hacen emerger "recovery" como concepto relevante.

Esto permite distinguir:

    similitud directa

de:

    relevancia contextual indirecta.


# 16. Esto proporciona dos tipos de distancia

## Distancia directa

    query ↔ chunk

"¿Qué chunk habla de algo parecido a mi pregunta?"

## Distancia conceptual expandida

    query
      ↓
    concepts
      ↓
    related concepts
      ↓
    chunks

"¿Qué otros chunks son relevantes después de explorar el espacio conceptual relacionado?"

La segunda es la que permite encontrar información satélite.


# 17. Cuánto se parece esto a investigaciones existentes

La respuesta es:

> Mucho en sus componentes, pero la combinación concreta es menos habitual.

Las líneas más relevantes son las siguientes.


# 18. LSI: el antecedente matemático más antiguo

Antes de los embeddings neuronales ya existía una formulación relacionada:

    term-document matrix
            ↓
           SVD
            ↓
    latent semantic space

Latent Semantic Indexing (LSI) transforma una matriz de términos-documentos mediante descomposición matricial para obtener un espacio semántico latente.

La enseñanza importante para nuestra idea es:

> **una matriz de unidades documentales × dimensiones semánticas puede utilizarse como representación geométrica para retrieval.**

Es uno de los antecedentes conceptuales más directos de la intuición de "la biblioteca tiene su propio espacio". :contentReference[oaicite:1]{index=1}


# 19. Embedded Topic Model (ETM)

Embedded Topic Model combina topic modelling con embeddings de palabras.

Aprende temas cuyas representaciones viven en un espacio semántico junto con las palabras, y produce topics interpretables. :contentReference[oaicite:2]{index=2}

Es relevante porque conecta:

    topics
    +
    embeddings
    +
    representación de documentos

Pero sigue siendo un modelo de topic modelling, no un sistema de retrieval conceptual iterativo.

## Cercanía con nuestra propuesta

    ALTA en:
        descubrimiento de dimensiones semánticas
        + embeddings

    BAJA en:
        retrieval iterativo
        + expansión de contexto


# 20. Top2Vec

Top2Vec es todavía más cercano conceptualmente.

Aprende conjuntamente:

    document vectors
    word vectors
    topic vectors

y utiliza la distancia entre ellos para identificar topics y recuperar documentos relacionados. Además descubre automáticamente el número de topics. :contentReference[oaicite:3]{index=3}

Es una referencia importante porque demuestra que es perfectamente viable pensar en:

    documentos
       ↕
    topics
       ↕
    palabras

dentro de un espacio semántico común.

Pero de nuevo:

> los topics de Top2Vec no constituyen exactamente un diccionario dinámico de conceptos utilizado como coordenadas explícitas de un retriever RAG iterativo.


# 21. BERTopic

BERTopic combina embeddings transformer, clustering y c-TF-IDF para crear representaciones interpretables de topics. Soporta además:

- distribución multilabel;
- topics jerárquicos;
- topics dinámicos;
- procesamiento incremental;
- guided topics;
- topic embeddings;
- visualización de similitud entre topics.

:contentReference[oaicite:4]{index=4}

Esto es muy relevante para la fase:

    corpus
       ↓
    descubrir estructura conceptual
       ↓
    representar unidades documentales mediante conceptos

Pero BERTopic está más orientado a:

    descubrir/organizar topics

que a:

    ejecutar retrieval iterativo para construir contexto de un agente.


# 22. Learned Sparse Retrieval: SPLADE

SPLADE probablemente sea uno de los antecedentes técnicos más importantes.

SPLADE representa documentos y consultas mediante vectores dispersos sobre un vocabulario y asigna pesos continuos a los términos. Estos vectores pueden aprovechar índices invertidos tradicionales. :contentReference[oaicite:5]{index=5}

Esto se parece mucho a:

    concepto_1 = 0
    concepto_2 = 0.83
    concepto_3 = 1.17
    ...

en vez de utilizar un embedding denso.

Ventajas:

    interpretabilidad parcial
    sparsity
    eficiencia
    compatibilidad con inverted indexes
    expansión semántica

Pero las dimensiones de SPLADE siguen estando ligadas al vocabulario del modelo.


# 23. DeepCT

DeepCT aprende un peso contextual para cada término de un documento:

    term → importance weight

La representación puede almacenarse en un índice invertido para retrieval eficiente. :contentReference[oaicite:6]{index=6}

Es conceptualmente relevante porque demuestra algo importante para nuestro diseño:

> **el valor asociado a una dimensión puede ser continuo y aprendido específicamente para representar la importancia contextual de esa dimensión.**

Otra pieza de nuestra arquitectura.


# 24. DyVo: Dynamic Vocabulary

DyVo es especialmente cercano.

En vez de limitar SPLADE al vocabulario del transformer, introduce un vocabulario dinámico basado en entidades/conceptos de Wikipedia.

Produce pesos para:

    wordpieces
    +
    entities/concepts

y combina ambas representaciones para hacer retrieval sparse. :contentReference[oaicite:7]{index=7}

La idea importante:

> **las dimensiones de una representación sparse no tienen por qué estar limitadas al tokenizer del modelo.**

Esto se acerca muchísimo a la idea de:

    dimensiones = conceptos del dominio/corpus.


# 25. SAE-SPLADE — probablemente la referencia MÁS cercana

En 2026 aparece:

**From Tokens to Concepts: Leveraging SAE for SPLADE**

de Yuxuan Zong, Mathias Vast, Basile Van Cooten, Laure Soulier y Benjamin Piwowarski.

La propuesta explícita es:

    SPLADE tradicional:
        espacio = vocabulario de tokens

    SAE-SPLADE:
        espacio = conceptos semánticos latentes
                   aprendidos con Sparse Autoencoders

El trabajo estudia precisamente cómo reemplazar el vocabulario del backbone por un espacio latente de conceptos semánticos y obtiene rendimiento comparable a SPLADE con mejoras de eficiencia. :contentReference[oaicite:8]{index=8}

Hay además implementación pública con código para:

    SAE pretraining
    SAE-SPLADE finetuning
    indexing
    evaluation

:contentReference[oaicite:9]{index=9}

### Importancia para nuestra idea

MUY ALTA.

Es probablemente la investigación que más directamente demuestra que:

> "las dimensiones de un retriever sparse pueden ser conceptos y no necesariamente tokens".

Sin embargo, su objetivo principal es retrieval, no construir una arquitectura completa de:

    conceptos
      ↓
    chunks
      ↓
    expansión conceptual iterativa
      ↓
    contexto global.


# 26. Sparse Autoencoders

La línea de Sparse Autoencoders es conceptualmente relevante porque intenta descomponer las representaciones neuronales en features dispersas e interpretables.

Trabajos como Cunningham et al. y Anthropic muestran que pueden aparecer features relativamente monosemánticas y que al aumentar el tamaño del diccionario aparecen distintas "resoluciones" de conceptos. :contentReference[oaicite:10]{index=10}

Esto conecta de forma bastante directa con la idea:

    espacio conceptual pequeño
       ↓
    conceptos generales

    espacio conceptual mayor
       ↓
    conceptos más específicos

Es decir:

> **el espacio conceptual puede tener resolución variable.**

Esto podría ser muy importante para construir una taxonomía de conceptos que evolucione con el corpus.


# 27. Concept Bottleneck Models Without Predefined Concepts

Schrodi et al. presentan un sistema de concept bottleneck en el que los conceptos no están predefinidos, sino descubiertos automáticamente, y además introducen selección dependiente de la entrada para utilizar sólo una pequeña parte de los conceptos. :contentReference[oaicite:11]{index=11}

Aunque este trabajo está orientado a visión/clasificación y no a RAG, es muy relevante conceptualmente:

    input
      ↓
    discover/select concepts
      ↓
    operate in interpretable concept space

La idea de:

> no fijar previamente todos los conceptos y activar sólo un subconjunto relevante

es muy próxima a la arquitectura que estamos imaginando.


# 28. Query Expansion

Aquí hay una segunda gran familia que afecta a la fase de retrieval.

Query Expansion lleva décadas estudiando:

    query
      ↓
    términos/conceptos relacionados
      ↓
    nueva query
      ↓
    retrieval

La literatura clásica incluye relevance feedback, coocurrencia de términos, expansión basada en corpus y expansión semántica. :contentReference[oaicite:12]{index=12}

Una revisión reciente confirma que la expansión semántica sigue siendo una línea activa de investigación. :contentReference[oaicite:13]{index=13}

Esto es muy importante porque nuestra:

    expansión conceptual iterativa

es, en parte, una versión moderna y semántica de esta idea.


# 29. Query2doc

Query2doc genera un pseudo-documento utilizando un LLM y lo usa para expandir la consulta antes del retrieval.

El enfoque mejora tanto sistemas sparse como dense. :contentReference[oaicite:14]{index=14}

Es relevante porque demuestra que:

> la pregunta inicial puede ser insuficiente y conviene crear una representación intermedia más rica antes de recuperar.

Nuestro enfoque sería diferente:

    query
      ↓
    conceptos
      ↓
    espacio conceptual del corpus

en vez de:

    query
      ↓
    pseudo-document
      ↓
    embedding
      ↓
    retrieval.


# 30. HyDE

HyDE hace algo parecido desde otra perspectiva.

A partir de la pregunta genera un documento hipotético y utiliza el embedding de ese documento para encontrar documentos reales cercanos. :contentReference[oaicite:15]{index=15}

La relación con nuestra propuesta es:

> utilizar una representación intermedia de la intención del usuario para mejorar el retrieval.

La diferencia es que nosotros proponemos que esa representación intermedia sea:

    conjunto de conceptos + pesos

en lugar de:

    documento hipotético.


# 31. GraphRAG de Microsoft

GraphRAG realiza extracción de entidades y relaciones, clustering jerárquico y generación de resúmenes de comunidades.

El objetivo es permitir retrieval local y global y capturar dependencias entre múltiples partes del corpus. :contentReference[oaicite:16]{index=16}

Esto es muy relevante para nuestro problema porque ataca exactamente la limitación:

> "los chunks aislados no capturan la estructura global del corpus."

Sin embargo, GraphRAG se apoya en:

    entidades
    +
    relaciones explícitas
    +
    comunidades
    +
    summaries

mientras que nuestra idea intenta apoyarse más fuertemente en:

    conceptos
    +
    pesos
    +
    asociaciones derivadas.

Esto evita tener que definir una ontología exhaustiva de tipos de arista.


# 32. DRIFT Search

DRIFT amplía GraphRAG combinando búsqueda local y global y realizando una exploración más flexible del grafo para responder preguntas detalladas sin pagar siempre el coste de una búsqueda global completa. :contentReference[oaicite:17]{index=17}

Conceptualmente se acerca mucho a nuestra idea de:

    retrieval inicial
       ↓
    descubrir estructura relevante
       ↓
    expandir
       ↓
    continuar hasta conseguir contexto suficiente.

Es una referencia especialmente interesante para estudiar la lógica de "cuánto explorar".


# 33. LightRAG

LightRAG combina representación vectorial y estructura de grafo y utiliza recuperación a diferentes niveles de granularidad. Además incorpora actualización incremental del conocimiento. :contentReference[oaicite:18]{index=18}

Es interesante porque se acerca a la filosofía:

    vector retrieval
         +
    estructuración semántica
         +
    navegación

en vez de usar exclusivamente similarity search.


# 34. RAPTOR

RAPTOR construye una jerarquía recursiva mediante:

    chunks
      ↓
    embeddings
      ↓
    clustering
      ↓
    summaries
      ↓
    nuevos niveles

y recupera sobre los diferentes niveles del árbol. :contentReference[oaicite:19]{index=19}

Esto es muy relevante para tu idea de "bibliotecaria":

    chunk
      ↓
    sección
      ↓
    capítulo
      ↓
    resumen
      ↓
    libro

RAPTOR no construye nuestro espacio conceptual explícito, pero resuelve una dimensión complementaria:

> **la jerarquía de abstracción.**

Por tanto:

    concept space + RAPTOR-like hierarchy

es una combinación especialmente interesante.


# 35. GRAG

GRAG intenta recuperar subgrafos textuales en lugar de documentos aislados y combinar información topológica y textual. :contentReference[oaicite:20]{index=20}

Esto aproxima la idea de:

    "no quiero un chunk aislado;
     quiero una región conectada del corpus."


# 36. HyperGraphRAG

HyperGraphRAG señala una limitación de los grafos tradicionales:

    A ─── B

sólo representa relaciones binarias.

Pero existen hechos que involucran:

    A
    B
    C
    D

simultáneamente.

Propone hiper-aristas capaces de representar relaciones n-arias y desarrolla construcción, retrieval y generación sobre esa estructura. Fue presentado en NeurIPS 2025. :contentReference[oaicite:21]{index=21}

Esto confirma algo importante:

> la comunidad está buscando estructuras más ricas que simples pares de entidades.


# 37. Hyper-RAG y HyCE-RAG

En 2026 la investigación ha avanzado aún más hacia retrieval basado en hipergráficos.

Hyper-RAG combina correlaciones binarias y de orden superior para retrieval basado en conocimiento estructurado. :contentReference[oaicite:22]{index=22}

HyCE-RAG construye un hipergráfico de evidencia dependiente de la consulta y utiliza propagación de confianza para seleccionar y ensamblar cadenas de evidencia en preguntas multi-hop. :contentReference[oaicite:23]{index=23}

Esto es bastante cercano a:

    recuperar
       ↓
    descubrir conexiones
       ↓
    construir una cadena de evidencia
       ↓
    ampliar contexto

pero vuelve a introducir estructura relacional explícita.


# 38. HGRAG y la idea de "shared entities"

Un trabajo de AAAI 2026 propone un hipergráfico en el que:

    entidades = nodos
    pasajes = hiper-aristas

de modo que varios pasajes quedan conectados porque comparten entidades.

Después combina:

    similitud de entidades
    +
    similitud de pasajes
    +
    difusión sobre el hipergráfico.

Además reporta una mejora de eficiencia de 6× frente a determinadas alternativas. :contentReference[oaicite:24]{index=24}

Conceptualmente está sorprendentemente cerca de nuestra idea:

> **la estructura de relaciones puede derivarse del hecho de compartir unidades semánticas.**

La diferencia es que ellos utilizan entidades e hipergráficos; nosotros estamos pensando en un diccionario conceptual mucho más general.


# 39. EHRAG

EHRAG (ACL 2026) combina construcción de hipergráficos con retrieval estructural-semántico, topic-aware scoring y Personalized PageRank para recuperar documentos relevantes. :contentReference[oaicite:25]{index=25}

Es particularmente interesante porque mezcla:

    semántica
    +
    topics
    +
    estructura
    +
    difusión

que es muy cercano a la arquitectura que estamos describiendo.


# 40. PanoramaRAG

PanoramaRAG, también de 2026, parte de un problema importante:

> los grafos locales pueden encontrar relaciones pero perder el contexto global del corpus.

Propone una especie de "panorama" del corpus para guiar tanto el refinamiento de la consulta como el retrieval. :contentReference[oaicite:26]{index=26}

Esto es muy relevante para nuestra propuesta porque la pregunta fundamental es:

> ¿cómo descubrir qué conocimiento adicional necesito antes de saber exactamente qué buscar?


# 41. LEDGER

LEDGER, ACL 2026, construye grafos ligeros de dependencias semánticas y jerarquías estructurales entre elementos documentales y utiliza traversal para recuperar sólo el contexto afectado por una operación. Reporta una reducción muy importante del contexto procesado frente a alternativas de contexto completo. :contentReference[oaicite:27]{index=27}

Es un buen ejemplo de algo que se acerca a:

    estructura + dependencias + retrieval selectivo

aunque el problema que resuelve es edición documental, no RAG general.


# 42. Lo más interesante: la literatura está convergiendo

Podemos agrupar las investigaciones encontradas en cinco familias.

## Familia A — Espacio conceptual

    ETM
    Top2Vec
    BERTopic
    Concept Bottleneck
    SAE

Objetivo:

    descubrir / representar conceptos


## Familia B — Sparse semantic retrieval

    SPLADE
    SPLADE++
    DeepCT
    DyVo
    SAE-SPLADE

Objetivo:

    representar documentos y queries
    con dimensiones interpretables/sparse


## Familia C — Expansión de query

    Query Expansion clásico
    Query2doc
    HyDE

Objetivo:

    enriquecer la consulta


## Familia D — Retrieval estructurado

    GraphRAG
    DRIFT
    LightRAG
    GRAG
    HyperGraphRAG
    Hyper-RAG
    HGRAG
    HyCE-RAG

Objetivo:

    recuperar información conectada


## Familia E — Estructura jerárquica

    RAPTOR
    hierarchical RAG
    community-based retrieval

Objetivo:

    recuperar contexto a diferentes niveles de abstracción.


# 43. Nuestra propuesta se situaría en la intersección

La arquitectura buscada sería aproximadamente:

                       CORPUS
                         │
                         ▼
                 análisis semántico
                         │
          ┌──────────────┼──────────────┐
          │              │              │
       CONCEPTOS      ESTRUCTURA    EMBEDDINGS
          │              │              │
          ▼              ▼              ▼
   diccionario       árbol/documento  concepto embeddings
   conceptual
          │
          ▼
   MATRIZ CHUNK × CONCEPTO
          │
          ▼
  asociaciones concepto-concepto
  asociaciones chunk-chunk
          │
          └──────────────────────┐
                                 │
                              QUERY
                                 │
                                 ▼
                     conceptos de la pregunta
                                 │
                                 ▼
                       concept embeddings
                                 │
                                 ▼
                 conceptos del corpus más cercanos
                                 │
                                 ▼
                         chunks iniciales
                                 │
                                 ▼
                     conceptos descubiertos
                                 │
                                 ▼
                         expansión semántica
                                 │
                                 ▼
                         nuevos chunks
                                 │
                                 ▼
                            reiterar
                                 │
                                 ▼
                          CONTEXTO FINAL


# 44. El punto diferencial de la propuesta

La principal diferencia respecto a GraphRAG sería:

    GraphRAG:
        texto
          ↓
        entidades
          ↓
        relaciones explícitas
          ↓
        graph retrieval

Nuestra idea:

    texto
      ↓
    conceptos
      ↓
    activaciones conceptuales
      ↓
    chunk × concept matrix
      ↓
    relaciones emergentes por similitud/coocurrencia
      ↓
    concept-space traversal
      ↓
    retrieval iterativo


# 45. Y frente a un embedding convencional

Embedding normal:

    chunk → vector de dimensión fija

Propuesta:

    chunk → vector en espacio conceptual dinámico

Ejemplo:

    convencional:
        [0.13, -0.41, 0.77, ...]

    conceptual:
        [LLM=.91,
         training=.84,
         RAG=.12,
         fine-tuning=.76,
         nutrition=.02,
         injuries=.00]


# 46. Una propiedad muy importante: el espacio puede crecer

Inicialmente:

    1.000 conceptos

Después de incorporar nuevos documentos:

    1.300 conceptos

Después:

    1.800 conceptos

Por tanto:

    K = K(corpus)

en vez de:

    K = constante del modelo

Esto permite una semántica específica del dominio.

Por ejemplo, en una biblioteca médica aparecerían:

    biomarker
    immunotherapy
    EGFR
    staging
    ...

En una biblioteca de ciclismo:

    FTP
    TSS
    CTL
    ATL
    glycogen
    pacing
    overreaching
    ...

Las dimensiones del espacio se adaptan al dominio.


# 47. Pero hay un problema importante

No queremos que el sistema termine con:

    500.000 conceptos

porque entonces las dimensiones dejan de ser útiles.

Por tanto necesitamos resolver:

> ¿Cuándo dos conceptos son suficientemente parecidos como para fusionarlos?

Por ejemplo:

    "ajuste de modelo"
    "adaptación de modelo"
    "model adaptation"
    "fine-tuning"

podrían acabar como:

    C_173 = fine-tuning / model adaptation

Esto convierte la creación del diccionario conceptual en un problema fundamental de:

    discovery
    normalization
    deduplication
    hierarchy
    granularity control


# 48. La investigación de SAE sugiere además un problema interesante

Las features pueden "partirse" al aumentar el número de features del diccionario.

Eso significa que podemos imaginar:

    CONCEPTO GENERAL
           │
           ├── concepto específico
           ├── concepto específico
           └── concepto específico

Por ejemplo:

    training
       │
       ├── pretraining
       ├── fine-tuning
       ├── instruction tuning
       └── RLHF

Esto podría permitir que el espacio conceptual tenga:

    resolución gruesa
    +
    resolución fina

y que el retrieval pueda trabajar a diferentes resoluciones. :contentReference[oaicite:28]{index=28}


# 49. Una arquitectura práctica inicial

No sería necesario entrenar un modelo nuevo.

Se podría construir un prototipo con modelos existentes.

## Fase 1 — segmentación

    documentos
       ↓
    capítulos
       ↓
    secciones
       ↓
    chunks

## Fase 2 — descubrimiento conceptual

Para cada chunk:

    LLM / extractor
       ↓
    conceptos candidatos

## Fase 3 — normalización

    synonyms
    spelling
    variants
    hierarchy
    duplicates

## Fase 4 — embedding de conceptos

    concept → embedding

## Fase 5 — asignación conceptual

    chunk + concept
          ↓
    score ∈ [0,1]

Resultado:

    X[chunk, concept] = score


# 50. Fase de query

    pregunta
       ↓
    extracción de conceptos
       ↓
    embedding de conceptos
       ↓
    nearest concepts
       ↓
    seleccionar dimensiones activas
       ↓
    recuperar chunks desde X


# 51. Fase de expansión

    chunks recuperados
       ↓
    conceptos fuertes
       ↓
    conceptos relacionados
       ↓
    ranking de conceptos satélite
       ↓
    nuevos chunks
       ↓
    reranking
       ↓
    repetir


# 52. La expansión debería ser query-aware

No basta con decir:

    training
       ↔
    nutrition

porque pueden aparecer juntos en el corpus.

Lo interesante es:

    training + competition + 200 km
          ↓
    ¿qué conceptos adicionales suelen ser relevantes
    en este contexto?

Por tanto la asociación debe depender de la consulta.

Conceptualmente:

    relevance(concept_satellite | query, retrieved_chunks)

en lugar de:

    relevance(concept_satellite | corpus)


# 53. Esto evita uno de los problemas del simple co-occurence

Puede ocurrir:

    training ↔ physiology

muy frecuentemente.

Pero si la pregunta es:

    "¿cómo entreno para mejorar sprint?"

quizá:

    physiology

sea relevante pero:

    cadence
    neuromuscular power
    anaerobic capacity

sean mucho más relevantes.

La expansión debería estar condicionada por el contexto actual de búsqueda.


# 54. Criterio de parada

Un sistema de este tipo necesita saber cuándo detener la expansión.

Posibles señales:

    relevancia del nuevo concepto respecto a query
    relevancia de los nuevos chunks
    cantidad de información nueva
    redundancia
    coste de tokens
    cobertura conceptual

Una posible intuición:

    iteración 0:
        conceptos directamente preguntados

    iteración 1:
        conceptos fuertemente conectados

    iteración 2:
        conceptos útiles pero indirectos

    iteración 3:
        mayoría de ruido

La investigación futura aquí parece especialmente abierta.


# 55. Qué NO intentaría inicialmente

No intentaría construir un knowledge graph clásico con miles de relaciones:

    uses
    causes
    depends_on
    contains
    part_of
    derived_from
    etc.

Eso añade una ontología que quizá no necesitas.

Tampoco intentaría sustituir completamente el embedding denso.

Es más interesante combinar:

    embedding denso
    +
    espacio conceptual sparse
    +
    estructura jerárquica
    +
    expansión semántica


# 56. La combinación podría tener cuatro señales

Para un chunk candidato:

    Score =
        similitud embedding
        +
        similitud conceptual
        +
        relevancia por expansión
        +
        coherencia estructural

Por ejemplo:

    embedding similarity      0.82
    conceptual similarity     0.91
    expansion relevance       0.76
    hierarchy coherence       0.87

y producir un ranking conjunto.


# 57. Relación con GraphRAG: diferencia conceptual

GraphRAG pregunta esencialmente:

> "¿Qué entidades y relaciones están conectadas?"

Nuestra propuesta preguntaría:

> "¿Qué conceptos explican estos chunks y qué otras regiones del espacio conceptual deberían explorarse para responder esta pregunta?"

Esto puede ser más apropiado para corpus donde las relaciones exactas son difíciles de definir pero las asociaciones temáticas son fuertes.


# 58. Una posible evolución futura: transformar el espacio conceptual en un grafo

No es necesario empezar por el grafo.

Podemos hacer:

    X = chunk × concept

y después derivar:

    concept ↔ concept
    chunk ↔ chunk

a partir de:

    similitud
    coactivación
    coocurrencia
    estructura documental
    embeddings conceptuales

Sólo cuando resulte útil convertir esas asociaciones en una estructura explícita se construye el grafo.

Es decir:

    matriz conceptual
           ↓
    relaciones derivadas
           ↓
         grafo

y no necesariamente:

    texto
      ↓
    grafo explícito


# 59. Esto es particularmente importante para tu "bibliotecaria"

La arquitectura conceptual sería:

    ┌──────────────────────────────────┐
    │             BIBLIOTECA           │
    │                                  │
    │  documentos                      │
    │      ↓                           │
    │  estructura jerárquica           │
    │      ↓                           │
    │  conceptos                       │
    │      ↓                           │
    │  matriz CHUNK × CONCEPTO         │
    │      ↓                           │
    │  espacio semántico conceptual    │
    │      ↓                           │
    │  asociaciones entre conceptos    │
    │      ↓                           │
    │  navegación / expansión          │
    └──────────────────────────────────┘

El vector denso convencional se convierte en una señal adicional, no en el único mecanismo de memoria.


# 60. Grado de cercanía de las principales investigaciones

| Trabajo / proyecto | Conceptos explícitos | Dimensiones dinámicas | Sparse vector | Embeddings de conceptos | Expansión iterativa | Grafo/estructura | Cercanía |
|---|---:|---:|---:|---:|---:|---:|---|
| LSI | parcial | no | no | no | no | no | ★★★ |
| ETM | sí | sí/latent | no | sí | no | no | ★★★★ |
| Top2Vec | sí | sí | no | sí | no | no | ★★★★ |
| BERTopic | sí | sí | parcial | sí | no | jerarquía opcional | ★★★★ |
| DeepCT | términos | no | sí | no | no | no | ★★★ |
| SPLADE | términos | no | sí | no | no | no | ★★★★ |
| DyVo | conceptos/entidades | dinámica | sí | sí | no | no | ★★★★½ |
| SAE-SPLADE | conceptos latentes | según diccionario | sí | implícito | no | no | ★★★★★ |
| Query2doc | no | no | no | sí | expansión query | no | ★★★★ |
| HyDE | no | no | no | sí | expansión query | no | ★★★ |
| RAPTOR | no | no | no | sí | sí, jerárquico | árbol | ★★★★ |
| GraphRAG | entidades | corpus-dependent | no | sí | traversal/query | sí | ★★★★½ |
| DRIFT | entidades | corpus-dependent | no | sí | sí | sí | ★★★★½ |
| LightRAG | entidades/relaciones | sí | sí/parcial | sí | sí | sí | ★★★★½ |
| HyperGraphRAG | entidades/facts | corpus-dependent | no | sí | sí | hipergráfico | ★★★★ |
| HGRAG | entidades + pasajes | corpus-dependent | no | sí | sí | hipergráfico | ★★★★ |
| EHRAG | topics/entities | corpus-dependent | parcial | sí | sí | hipergráfico | ★★★★½ |
| PanoramaRAG | topics/global structure | corpus-dependent | no | sí | sí | grafo | ★★★★½ |

Las valoraciones de "cercanía" son una valoración conceptual de esta investigación, no resultados experimentales.


# 61. Referencias prioritarias para estudiar

## Prioridad 1 — leer inmediatamente

### SAE-SPLADE

"From Tokens to Concepts: Leveraging SAE for SPLADE", SIGIR 2026.

Es la referencia más directamente alineada con la hipótesis de sustituir dimensiones léxicas por conceptos semánticos.

Paper:
https://arxiv.org/abs/2604.21511

Código:
https://github.com/yzong12138/sae_splade

:contentReference[oaicite:29]{index=29}


### DyVo

"DyVo: Dynamic Vocabularies for Learned Sparse Retrieval with Entities"

Especialmente importante para la idea de vocabulario dinámico y dimensiones que no dependen exclusivamente del tokenizer.

:contentReference[oaicite:30]{index=30}


### SPLADE

"SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking"

"SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval"

Fundamental para entender cómo se puede utilizar una representación sparse con valores continuos y retrieval eficiente.

:contentReference[oaicite:31]{index=31}


# 62. Prioridad 2 — espacio conceptual

### Top2Vec

Especialmente importante por su idea de:

    document vectors
    +
    topic vectors
    +
    word vectors

dentro de un espacio compartido.

:contentReference[oaicite:32]{index=32}


### BERTopic

Especialmente interesante para:

    descubrimiento de conceptos
    clustering
    jerarquías
    embeddings de topics
    evolución del corpus

:contentReference[oaicite:33]{index=33}


### Embedded Topic Model

Especialmente relevante para conectar:

    topic space
    +
    embedding space

:contentReference[oaicite:34]{index=34}


# 63. Prioridad 3 — retrieval por expansión

### Query Expansion

Para comprender el antecedente teórico de:

    retrieval
      ↓
    nuevo conocimiento
      ↓
    expansión
      ↓
    retrieval

:contentReference[oaicite:35]{index=35}


### Query2doc

Para estudiar cómo un LLM puede enriquecer una consulta antes del retrieval.

:contentReference[oaicite:36]{index=36}


### HyDE

Para estudiar el uso de una representación intermedia generada a partir de la pregunta.

:contentReference[oaicite:37]{index=37}


# 64. Prioridad 4 — retrieval estructurado

### GraphRAG

https://www.microsoft.com/en-us/research/project/graphrag/

Especialmente importante para:

    indexación estructurada
    entidades
    relaciones
    comunidades
    global/local retrieval

:contentReference[oaicite:38]{index=38}


### DRIFT

Especialmente relevante para:

    query-aware traversal
    local + global retrieval
    control del coste de exploración

:contentReference[oaicite:39]{index=39}


### LightRAG

Especialmente relevante para:

    graph + vector
    retrieval multinivel
    actualización incremental

:contentReference[oaicite:40]{index=40}


# 65. Prioridad 5 — estructuras de mayor orden

### HyperGraphRAG

Muy interesante para estudiar qué ocurre cuando una relación no es simplemente:

    A ↔ B

sino:

    A + B + C + contexto

:contentReference[oaicite:41]{index=41}


### HGRAG

Particularmente interesante por representar:

    entidades = nodos
    pasajes = hiper-aristas

y recuperar mediante difusión combinando estructura y semántica. :contentReference[oaicite:42]{index=42}


### EHRAG

Interesante por combinar:

    topic-aware scoring
    +
    hybrid diffusion
    +
    graph structure

:contentReference[oaicite:43]{index=43}


### HyCE-RAG

Importante para estudiar:

    query-aware evidence graph
    +
    confidence propagation
    +
    multi-hop retrieval

:contentReference[oaicite:44]{index=44}


# 66. Qué parece faltar en conjunto

Después de revisar estas líneas, la hipótesis más interesante no parece ser:

> "crear otro GraphRAG".

Tampoco:

> "crear otro embedding".

La parte potencialmente diferencial sería combinar:

    1. descubrimiento automático de conceptos
    2. diccionario conceptual dinámico
    3. embedding semántico de conceptos
    4. matriz chunk × concepto
    5. retrieval vectorial dentro de ese espacio interpretable
    6. descubrimiento de conceptos satélite
    7. expansión iterativa query-aware
    8. jerarquía documental
    9. embedding denso como señal complementaria
    10. criterio de parada de la expansión


# 67. Una formulación compacta de la arquitectura

Puede definirse como:

    CORPUS
      ↓
    Concept Discovery
      ↓
    Concept Dictionary C
      ↓
    Concept Embeddings E(C)
      +
    Chunk–Concept Matrix X
      ↓
    Query Concept Mapping
      ↓
    Initial Concept Retrieval
      ↓
    Initial Chunk Retrieval
      ↓
    Concept Expansion
      ↓
    Chunk Expansion
      ↓
    Relevance / Novelty / Coverage Evaluation
      ↓
    Repeat
      ↓
    Hierarchical Context Assembly
      ↓
    LLM


# 68. La idea de "espacio semántico de la biblioteca"

La interpretación conceptual más interesante sería ésta:

Un embedding convencional dice:

> "Este chunk ocupa esta posición en el espacio aprendido por mi modelo."

Nuestro sistema dice:

> "Este chunk ocupa esta posición en el espacio semántico propio de esta biblioteca."

Por ejemplo:

    dimensión 1 = entrenamiento
    dimensión 2 = nutrición
    dimensión 3 = lesiones
    dimensión 4 = recuperación
    dimensión 5 = competición
    ...

Esto crea una especie de:

> **mapa semántico de la biblioteca.**

Y la pregunta del usuario entra primero en ese mapa antes de seleccionar documentación.


# 69. La hipótesis de investigación más interesante

La formularía así:

> **¿Puede un corpus documental ser representado mediante un espacio semántico sparse y dinámico cuyas dimensiones correspondan a conceptos descubiertos a partir del propio corpus, de forma que cada unidad documental tenga una representación interpretable en dicho espacio y que el retrieval pueda expandirse iterativamente a través de conceptos emergentes para recuperar contexto indirectamente relevante para la pregunta?**

Y añadiría una segunda hipótesis:

> **¿Mejora esta representación el recall contextual respecto a un RAG basado exclusivamente en embeddings densos, especialmente en preguntas que requieren información satélite o multi-hop que no aparece explícitamente en la query?**


# 70. El experimento que mejor probaría la idea

No hace falta empezar entrenando nada.

Construiría un prototipo sobre un corpus relativamente homogéneo.

Por ejemplo:

    20–50 libros
    5.000–20.000 chunks

Crear:

    X = chunk × concept

y probar tres sistemas:

### Sistema A — baseline

    dense embedding
    ↓
    vector search
    ↓
    top-K


### Sistema B

    dense embedding
    +
    concept vector
    ↓
    hybrid retrieval


### Sistema C

    dense embedding
    +
    concept vector
    +
    iterative concept expansion
    ↓
    final context


Y medir:

    Recall@K
    Context Recall
    Context Precision
    answer faithfulness
    answer completeness
    token cost
    number of retrieval iterations


# 71. La pregunta experimental clave

No mediría solamente:

> "¿Encuentra el documento correcto?"

También mediría:

> **"¿Encuentra todos los conocimientos necesarios para contestar correctamente?"**

Porque ahí está precisamente la motivación del sistema.

Un retrieval puede tener:

    top-5 muy buenos

y aun así ser insuficiente porque falta:

    nutrición
    lesiones
    recuperación

que son conocimientos necesarios pero no explícitos en la pregunta.


# 72. Evaluación especialmente interesante

Construir preguntas de tres clases:

### Tipo A — directo

    "¿Qué es el fine-tuning?"

El dense retrieval debería funcionar muy bien.

### Tipo B — multiconcepto

    "¿Cómo afecta el volumen de entrenamiento
     a la recuperación?"

Aquí el espacio conceptual puede ser especialmente útil.

### Tipo C — contexto satélite

    "Diseña un entrenamiento para una carrera de 200 km
     teniendo en cuenta mis características."

Aquí la pregunta deliberadamente no menciona:

    nutrición
    hidratación
    pacing
    lesiones
    recuperación

y se puede comprobar si el sistema las descubre.


# 73. Criterio de éxito

La demostración fuerte de la arquitectura sería algo así:

    Dense RAG:
        encuentra 8 chunks
        cubre 55% del conocimiento necesario

    Concept RAG:
        encuentra 10 chunks
        cubre 68%

    Iterative Concept RAG:
        encuentra 14 chunks
        cubre 91%

sin aumentar proporcionalmente el ruido ni los tokens enviados al LLM.


# 74. Conclusión

La intuición original tiene bastante fundamento.

No se trata simplemente de:

> "poner etiquetas a los chunks".

Lo interesante es interpretar esas etiquetas como un **espacio semántico explícito del corpus**.

Cada chunk se convierte entonces en un vector:

    chunk =
    [concept_1, concept_2, ..., concept_K]

donde cada coordenada tiene significado.

Los embeddings convencionales pueden utilizarse de forma complementaria para resolver:

    concepto de query
          ↓
    concepto del corpus

La matriz chunk × concepto permite después resolver:

    concepto
       ↓
    chunks

y, mediante comparación/coocurrencia, obtener:

    conceptos ↔ conceptos
    chunks ↔ chunks

Esto permite construir un retrieval que no sea puramente:

    QUERY → TOP-K

sino:

    QUERY
      ↓
    CONCEPTS
      ↓
    INITIAL CHUNKS
      ↓
    DISCOVERED CONCEPTS
      ↓
    NEW CHUNKS
      ↓
    NEW CONCEPTS
      ↓
    ...
      ↓
    SUFFICIENT CONTEXT

La literatura más cercana actualmente es:

    SAE-SPLADE
        ↓
    conceptos como dimensiones de retrieval sparse

    DyVo
        ↓
    vocabulario/conceptos dinámicos

    BERTopic / Top2Vec / ETM
        ↓
    espacios de topics/conceptos inducidos del corpus

    Query Expansion / Query2doc / HyDE
        ↓
    expansión de la intención de la query

    GraphRAG / DRIFT / LightRAG
        ↓
    exploración de estructura relacionada

    HyperGraphRAG / HGRAG / EHRAG / HyCE-RAG
        ↓
    recuperación multi-hop estructurada

    RAPTOR
        ↓
    jerarquía y abstracción documental

La observación más importante de esta investigación es que **SAE-SPLADE (SIGIR 2026) hace que la primera parte de la hipótesis deje de ser puramente especulativa**: ya existe un trabajo reciente que reemplaza explícitamente el espacio de tokens por un espacio de conceptos latentes aprendido para retrieval sparse. :contentReference[oaicite:45]{index=45}

Lo que parece menos cubierto por una solución única es la combinación:

    "concept-space retrieval"
             +
    "concept-to-query semantic mapping"
             +
    "query-aware concept expansion"
             +
    "iterative chunk retrieval"
             +
    "document hierarchy"

Ese punto de intersección es, a mi juicio, donde está la parte más interesante de tu idea y donde merecería hacer una revisión bibliográfica todavía más específica antes de diseñar un prototipo.