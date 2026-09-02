Sí. He tomado el archivo correcto como base. El **proyecto principal** no es ninguno de los papers que aparecen dentro del documento: es la arquitectura que se propone ahí, centrada en un **espacio conceptual dinámico e inducido por el corpus**, una matriz **chunk × concepto**, y un **retrieval iterativo guiado por conceptos descubiertos durante la propia búsqueda**. El documento formula precisamente esa combinación como hipótesis y diferencia entre los dos espacios: embedding de conceptos y representación chunk×concepto. 

Además, he contrastado las líneas citadas en el documento con sus publicaciones originales y documentación oficial. El resultado importante es bastante distinto de la conclusión superficial del análisis anterior.

# 1. Qué estoy tomando como proyecto principal

Lo voy a llamar **Dynamic Concept-Space RAG (DCS-RAG)** para poder compararlo con los demás.

Su núcleo tiene **10 piezas**:

1. Descubrimiento automático de conceptos a partir del corpus.
2. Diccionario conceptual propio del corpus.
3. Conceptos dinámicos: el número y granularidad pueden evolucionar.
4. Embedding independiente de cada concepto.
5. Matriz `X = chunks × concepts`.
6. Vector conceptual interpretable para cada chunk.
7. Mapeo de la query → conceptos del corpus.
8. Retrieval conceptual.
9. Expansión iterativa **query-aware**:
   `query → conceptos → chunks → nuevos conceptos → nuevos chunks → ...`
10. Combinación con embeddings densos y estructura documental, con un criterio de parada basado en relevancia, novedad, cobertura y coste.

Eso último es crucial: el documento no plantea simplemente "usar conceptos" sino **utilizar el espacio conceptual como superficie de navegación del corpus**. 

Y tampoco plantea necesariamente un knowledge graph tradicional: las relaciones pueden emerger de coactivación, similitud, coocurrencia y vecindad estructural en la matriz. 

---

# 2. La comparación que realmente importa

Hay cinco familias distintas mezcladas en el documento:

| Familia                 | Qué resuelve                                        |
| ----------------------- | --------------------------------------------------- |
| Topic/concept discovery | Descubrir dimensiones semánticas                    |
| Sparse retrieval        | Representar query/documentos con features dispersas |
| Query expansion         | Enriquecer la intención de la consulta              |
| Graph / hypergraph RAG  | Navegar relaciones entre piezas de conocimiento     |
| Hierarchical RAG        | Recuperar a distintos niveles de abstracción        |

**DCS-RAG intenta unir las cinco**, pero de una manera concreta que ninguna de ellas, por separado, reproduce.

---

# 3. Comparación proyecto por proyecto

## 3.1 LSI — **7/10**

**Qué hace**

Latent Semantic Indexing construye una matriz término-documento y utiliza una factorización para proyectarla a un espacio semántico latente. Su objetivo original era explotar estructura semántica implícita de las asociaciones término-documento para mejorar retrieval. ([asistdl.onlinelibrary.wiley.com][1])

**Similitudes con DCS-RAG**

Muy profundas a nivel conceptual:

`unidad documental × dimensión semántica → representación geométrica`

Eso está muy cerca de:

`chunk × concept → vector conceptual`.

Además, LSI introduce una idea fundamental que sigue siendo válida:

> el corpus puede definir una geometría semántica propia.

**Diferencias**

LSI:

* utiliza dimensiones latentes matemáticas;
* no necesita conceptos interpretables;
* las dimensiones no equivalen a conceptos nombrables;
* es una proyección estática;
* no hace expansión durante retrieval;
* no tiene query-aware concept discovery.

DCS-RAG sustituye precisamente ese espacio latente opaco por dimensiones semánticas interpretables y navegables.

**Mi valoración:** **7/10**.

Es un antecedente conceptual importantísimo, pero no una solución equivalente.

---

# 4. Embedded Topic Model — **7,5/10**

ETM combina topic modelling y embeddings: los topics y palabras viven en un espacio semántico común y los documentos se modelan mediante distribución sobre topics. ([ACL Anthology][2])

### Similitudes

Aquí aparece una pieza muy próxima:

`documento → distribución sobre dimensiones semánticas`

y además:

`topic ↔ embedding`.

Eso se parece bastante a:

`chunk → pesos sobre conceptos`

y

`concepto → embedding`.

### Diferencias

ETM sigue siendo fundamentalmente **topic modelling generativo**.

No intenta resolver:

`query → concept mapping → retrieval → concept discovery → retrieval`

ni trata los topics como una infraestructura de navegación de RAG.

Tampoco está pensado para que aparezca durante retrieval un concepto satélite y éste desencadene una nueva búsqueda.

**Cercanía: 7,5/10**

---

# 5. Top2Vec — **8/10**

Top2Vec es uno de los antecedentes más interesantes del documento. Aprende conjuntamente **document vectors, word vectors y topic vectors**, y utiliza su proximidad semántica; además determina automáticamente el número de topics. ([arXiv][3])

### Similitud

Su estructura conceptual es casi:

`documents ↔ topics ↔ words`

DCS-RAG:

`chunks ↔ concepts ↔ query concepts`.

Hay una correspondencia muy clara.

Además Top2Vec elimina la necesidad de fijar previamente el número de topics, lo que conecta directamente con la idea del documento de que `K = K(corpus)`.

### Diferencia decisiva

Top2Vec responde:

> "¿Qué temas existen y qué documentos están asociados a ellos?"

DCS-RAG pregunta:

> "Dada esta query, ¿qué conceptos debo activar y qué regiones del espacio conceptual debo explorar a continuación?"

Top2Vec es fundamentalmente **discovery + representation**.

DCS-RAG quiere hacer **discovery + representation + retrieval traversal**.

**Cercanía: 8/10**

---

# 6. BERTopic — **8/10**

BERTopic genera embeddings documentales, los agrupa y posteriormente genera representaciones interpretables mediante c-TF-IDF. ([arXiv][4])

### Similitudes

Muy alta coincidencia en:

* descubrimiento automático de estructura;
* conceptos/topics emergentes;
* representación interpretable;
* clustering;
* posibilidad de trabajar jerárquicamente;
* embeddings de topics.

El documento identifica correctamente esta línea como muy cercana al componente de **construcción del espacio conceptual**. 

### Diferencia

BERTopic no convierte ese espacio en un **retriever iterativo conceptual**.

No existe:

`query → concepts → retrieve → discover concepts → retrieve`

como mecanismo central.

**Cercanía: 8/10**

---

# 7. DeepCT — **6,5/10**

DeepCT aprende pesos contextuales para términos utilizando representaciones contextualizadas de BERT y produce vectores ponderados que pueden almacenarse en índices invertidos. ([arXiv][5])

### Similitud

Aquí aparece una pieza muy importante de DCS-RAG:

`feature → importance weight`

en lugar de:

`feature → 0/1`.

Esto respalda la idea del documento de que una dimensión conceptual puede tener un valor continuo.

### Diferencia

La dimensión sigue siendo **un término**.

DCS-RAG quiere que sea:

`concepto semántico`.

Además DeepCT no crea un espacio conceptual dinámico ni realiza navegación conceptual.

**Cercanía: 6,5/10**

---

# 8. SPLADE / SPLADE++ — **8,5/10**

Aquí estamos ya en una zona muy importante.

SPLADE utiliza representaciones sparse con pesos continuos y aprovecha las propiedades de los índices invertidos. ([arXiv][6])

### Similitudes

DCS-RAG comparte prácticamente la filosofía:

`texto → sparse semantic representation`

y:

`feature_i → weight_i`.

También existe una fuerte analogía entre:

`SPLADE query vector`

y

`DCS-RAG query concept vector`.

### Diferencia fundamental

En SPLADE, las dimensiones proceden esencialmente del **vocabulario del backbone**.

Es decir:

`dimensión = token/term`.

En DCS-RAG:

`dimensión = concepto descubierto por el corpus`.

Esto no es un matiz. Es una diferencia arquitectónica esencial.

Y aquí entra el trabajo más importante de toda la búsqueda.

---

# 9. DyVo — **9/10**

DyVo extiende learned sparse retrieval incorporando un vocabulario dinámico de entidades/conceptos de Wikipedia además del vocabulario de wordpieces. El modelo combina pesos de wordpieces y entidades y los indexa con un inverted index. ([arXiv][7])

### Por qué es tan cercano

La idea:

> las dimensiones sparse pueden ser algo más rico que los tokens del tokenizer

es prácticamente una de las hipótesis fundamentales de DCS-RAG.

DCS-RAG dice:

`dimensiones = conceptos del corpus`.

DyVo demuestra:

`dimensiones = vocabulario dinámico + entidades`.

### Diferencias

DyVo sigue estando centrado en:

* entidades;
* Wikipedia;
* learned sparse retrieval.

No intenta construir un **espacio conceptual específico de cada corpus** donde cualquier concepto semántico pueda ser una dimensión.

Tampoco propone la fase:

`concept discovered → new retrieval → further concepts`.

**Cercanía: 9/10**

Es uno de los papers que yo estudiaría en profundidad antes de escribir una sola línea de DCS-RAG.

---

# 10. SAE-SPLADE — **9,5/10**

Este es, efectivamente, el competidor conceptual más importante.

El trabajo **From Tokens to Concepts: Leveraging SAE for SPLADE**, publicado en 2026, sustituye el espacio basado en vocabulario por un espacio de **conceptos semánticos latentes aprendido mediante Sparse Autoencoders**. Sus autores reportan rendimiento comparable a SPLADE con mejoras de eficiencia. ([arXiv][8])

### Similitud

Aquí la coincidencia es enorme:

DCS-RAG:

`chunk → [concept_1, concept_2, ..., concept_K]`

SAE-SPLADE:

`document/query → sparse latent concepts`.

Y el objetivo explícito de SAE-SPLADE es precisamente pasar:

`tokens → concepts`.

Eso destruye cualquier afirmación ingenua del tipo:

> "usar conceptos como dimensiones de un retriever sparse es completamente nuevo."

**No lo es.**

### Pero hay una diferencia muy importante

SAE-SPLADE aprende un **diccionario latente de features del modelo**.

DCS-RAG quiere:

`concept dictionary = induced by the corpus`.

Además DCS-RAG añade:

`concept embeddings`

y, sobre todo:

`concept → chunk → discovered concept → new chunk`.

SAE-SPLADE no es una arquitectura de navegación conceptual iterativa.

### Mi valoración

**9,5/10**

Esta es la referencia que más reduce la novedad de la primera mitad de la propuesta, pero **no elimina la posible novedad de la arquitectura completa**.

---

# 11. Sparse Autoencoders — **6,5/10**

Los trabajos de SAE muestran que una representación latente puede descomponerse en features dispersas más interpretables que los neurones originales. ([arXiv][9])

Anthropic mostró además el fenómeno de **feature splitting**: al aumentar el diccionario aparecen features más específicas, lo que proporciona precisamente la idea de diferentes niveles de resolución semántica. ([Transformer Circuits][10])

### Similitud

Muy importante para:

* diccionario dinámico;
* sparsity;
* interpretabilidad;
* granularidad;
* jerarquía de conceptos.

### Diferencia

SAE intenta descomponer las **activaciones internas de un modelo**.

DCS-RAG intenta representar el **corpus documental**.

Son objetos distintos.

**Cercanía: 6,5/10**

---

# 12. Concept Bottleneck Models Without Predefined Concepts — **7/10**

El trabajo de Schrodi et al. elimina la necesidad de un conjunto de conceptos previamente definido, usando discovery automático y selección dependiente de la entrada. ([arXiv][11])

### Similitud

La coincidencia con DCS-RAG en la filosofía es sorprendente:

`input → discover/select concepts → operate in concept space`.

Especialmente:

> no necesitas activar todo el diccionario para cada input.

Eso coincide con el concepto de **active concept subset** del documento.

### Diferencia

Es una arquitectura de clasificación/visión, no retrieval.

No existe:

`concept → corpus → document retrieval`.

**Cercanía: 7/10**

Es una referencia conceptual más que arquitectónica.

---

# 13. Query Expansion clásico — **8,5/10**

Aquí hay una conexión histórica fundamental.

Query Expansion lleva décadas haciendo:

`query → related terms → expanded query → retrieval`.

La literatura cubre relevance feedback, expansión basada en corpus, coocurrencia y otras variantes. ([ScienceDirect][12])

### Similitud

DCS-RAG está claramente dentro de esta familia.

Su:

`concept expansion`

es esencialmente una evolución semántica de:

`query expansion`.

### Diferencia

La propuesta no expande simplemente la **query textual**.

Expande un estado en un **espacio conceptual del corpus**.

Esto cambia:

`query reformulation`

por:

`semantic-space traversal`.

### Cercanía

**8,5/10**

---

# 14. Query2doc — **7/10**

Query2doc genera pseudo-documentos con un LLM para enriquecer la consulta y mejorar el retrieval. Ha demostrado mejoras tanto sobre sparse como sobre dense retrieval. ([ACL Anthology][13])

### Similitud

Ambos reconocen:

> la query original puede ser demasiado pobre.

Ambos construyen una representación intermedia más rica antes de recuperar.

### Diferencia

Query2doc:

`query → pseudo-document → retrieval`.

DCS-RAG:

`query → concepts → concept-space → chunks`.

Y, además, DCS-RAG sigue expandiéndose **después del primer retrieval**.

**Cercanía: 7/10**

---

# 15. HyDE — **6,5/10**

HyDE genera un documento hipotético, lo embebe y usa el embedding para encontrar documentos reales. ([ACL Anthology][14])

### Similitud

Ambos usan una **representación intermedia** para romper la insuficiencia de la query original.

### Diferencia

HyDE crea:

`hypothetical document`.

DCS-RAG crea:

`weighted concept vector`.

Y DCS-RAG puede navegar iterativamente.

**Cercanía: 6,5/10**

---

# 16. GraphRAG — **8,5/10**

GraphRAG construye una estructura basada en entidades y relaciones, identifica comunidades y genera community reports para soportar retrieval local y global. ([Microsoft][15])

### Similitud

Esta es una coincidencia de nivel arquitectónico:

`chunks aislados → estructura global → navegación → retrieval`.

DCS-RAG intenta solucionar precisamente el problema de que los chunks aislados no contienen todo el contexto.

### Diferencia

GraphRAG:

`entity + relation + community`.

DCS-RAG:

`concept + weight + association`.

Y esta diferencia es fundamental.

GraphRAG necesita crear relaciones explícitas.

DCS-RAG puede derivar las relaciones estadísticamente:

`concept coactivation`.

### Cercanía

**8,5/10**

No porque utilice la misma estructura, sino porque intenta resolver prácticamente el mismo **problema de retrieval global**.

---

# 17. DRIFT Search — **9/10**

Aquí encontramos una coincidencia todavía más fuerte.

DRIFT amplía GraphRAG mediante una lógica de:

`global context → local search → follow-up questions → further retrieval`.

Además utiliza una señal de confianza para determinar si debe continuar la expansión. ([Microsoft en GitHub][16])

### Esto se parece muchísimo a DCS-RAG

DCS:

`query → initial concepts → chunks → new concepts → new chunks → stop`.

DRIFT:

`query → broad context → follow-up questions → local retrieval → further exploration`.

La diferencia está en **qué se propaga**.

DRIFT propaga:

`questions / graph traversal`.

DCS-RAG propaga:

`concepts`.

### Cercanía

**9/10**

Este es probablemente el trabajo más importante para estudiar la **dinámica de expansión**, mientras SAE-SPLADE es el más importante para el **espacio de representación**.

---

# 18. LightRAG — **9/10**

LightRAG combina knowledge graph y embeddings, con retrieval en distintos niveles y soporte de actualización incremental. ([arXiv][17])

### Similitudes

Prácticamente comparte la filosofía:

`vector retrieval + semantic structure + navigation`.

Y su diseño pretende superar la fragmentación producida por el retrieval plano.

### Diferencia

LightRAG trabaja con:

`entities + relationships + KG`.

DCS-RAG:

`concepts + weights + implicit associations`.

### Cercanía

**9/10**

Porque está mucho más cerca de la arquitectura final que de una pieza aislada.

---

# 19. RAPTOR — **8/10**

RAPTOR construye recursivamente una jerarquía de chunks, embeddings, clustering y summaries, permitiendo recuperar diferentes niveles de abstracción. ([arXiv][18])

### Similitud

DCS-RAG también quiere:

`chunk → section → chapter → document`

y recuperar contexto a distintos niveles.

### Diferencia

RAPTOR organiza por **abstracción documental**.

DCS-RAG organiza por **espacio conceptual**.

Por eso son mucho más complementarios que competidores.

La combinación:

`RAPTOR hierarchy + concept-space retrieval`

es especialmente interesante.

**Cercanía: 8/10**

---

# 20. GRAG — **8/10**

GRAG recupera **subgrafos textuales** en lugar de documentos aislados y combina información textual y topológica. ([arXiv][19])

### Similitud

Coincide con una intuición central del proyecto:

> no quiero simplemente el chunk más próximo; quiero la región relevante del corpus.

### Diferencia

GRAG presupone una estructura de grafo.

DCS-RAG puede llegar a esa estructura **después**, derivándola de la matriz conceptual.

**Cercanía: 8/10**

---

# 21. HyperGraphRAG — **8,5/10**

HyperGraphRAG usa hiper-aristas para representar relaciones n-arias y construye sobre ellas retrieval y generación. Fue presentado en NeurIPS 2025. ([Proceedings NeurIPS][20])

### Similitudes

Ambos dicen:

> las relaciones relevantes pueden involucrar más de dos piezas.

Y ambos quieren recuperar **estructuras de conocimiento**, no simplemente chunks.

### Diferencia

HyperGraphRAG representa:

`fact = entities + relation`

DCS-RAG:

`association = concepts jointly activated`.

La representación es, por tanto, radicalmente distinta.

**Cercanía: 8,5/10**

---

# 22. Hyper-RAG — **7,5/10**

Hyper-RAG extiende el paradigma hypergraph para representar correlaciones binarias y de orden superior y está orientado a reducir alucinaciones mediante retrieval estructurado. La versión de 2026 aparece en *Nature Communications*. ([DOI][21])

### Similitud

La misma intuición:

`knowledge is not just a set of independent chunks`.

### Diferencia

Hyper-RAG utiliza estructura relacional explícita.

DCS-RAG intenta retrasar esa decisión y trabajar primero con el espacio conceptual.

**Cercanía: 7,5/10**

---

# 23. HGRAG — **9/10**

Aquí hay una coincidencia muy interesante.

HGRAG construye:

`entities = nodes`

`passages = hyperedges`

y utiliza difusión que combina similitud de entidades y de pasajes. El trabajo apareció en AAAI 2026 y reporta una mejora de eficiencia de alrededor de 6× frente a las alternativas evaluadas. ([ojs.aaai.org][22])

### La similitud realmente importante

La estructura:

`pasaje ↔ múltiples entidades`

es matemáticamente cercana a:

`chunk ↔ múltiples concepts`.

Es decir, ambos pueden verse como una estructura de **incidencia bipartita**.

### Diferencia

HGRAG:

`entity graph/hypergraph`.

DCS-RAG:

`concept incidence matrix`.

Pero esta diferencia es muchísimo menor de lo que parece.

**Cercanía: 9/10**

Este trabajo merece entrar en la lista de lectura prioritaria.

---

# 24. EHRAG — **9/10**

EHRAG es especialmente relevante porque intenta combinar:

* estructura;
* semántica;
* clustering de embeddings;
* difusión;
* topic-aware scoring;
* Personalized PageRank.

Su hipergráfico contiene tanto relaciones estructurales como relaciones semánticas. ([ACL Anthology][23])

### Por qué es peligroso para la afirmación de novedad

Porque su filosofía se aproxima muchísimo a:

`semantic representation + structure + propagation + retrieval`.

La diferencia está en que EHRAG utiliza **hyperedges sobre entidades** mientras DCS-RAG propone un **espacio conceptual explícito**.

### Cercanía

**9/10**

Es uno de los trabajos que más hay que analizar antes de afirmar que la arquitectura completa es novedosa.

---

# 25. HyCE-RAG — **9,5/10**

HyCE-RAG, publicado en 2026, construye un **query-aware evidence hypergraph**, realiza propagación de confianza y ensambla cadenas de evidencia para multi-hop QA. ([arXiv][24])

Aquí tenemos:

`query`

↓

`evidence structure`

↓

`propagation`

↓

`evidence chain`

↓

`new context`.

DCS-RAG:

`query`

↓

`concepts`

↓

`chunks`

↓

`new concepts`

↓

`new chunks`.

### Diferencia

HyCE-RAG propaga sobre **evidence hypergraphs**.

DCS-RAG propaga sobre un **concept space**.

Pero ambas arquitecturas comparten una lógica profunda:

> la búsqueda no termina en el primer conjunto recuperado; la evidencia encontrada modifica el espacio de búsqueda posterior.

**Cercanía: 9,5/10**

Es probablemente la referencia más importante de 2026 para la parte de **retrieval iterativo query-aware**.

---

# 26. PanoramaRAG — **9/10**

PanoramaRAG incorpora un "panorama" global del corpus para guiar tanto query refinement como retrieval y solucionar el problema de que los grafos locales pierdan contexto global. ([ACL Anthology][25])

### Similitud

Esto se aproxima mucho a:

> "¿Cómo saber qué conocimiento adicional necesito antes de saber exactamente qué buscar?"

El "panorama" funciona como una representación global que guía la búsqueda.

### Diferencia

PanoramaRAG utiliza una estructura de grafo global.

DCS-RAG plantea:

`global semantic concept space`.

### Cercanía

**9/10**

---

# 27. LEDGER — **7/10**

LEDGER construye grafos ligeros de dependencias semánticas y jerarquías estructurales y usa traversal para recuperar sólo el contexto afectado. En 2026 reporta una reducción del uso de tokens del 85% en su tarea de edición documental. ([ACL Anthology][26])

### Similitud

Tiene una idea muy próxima:

`structure → affected region → selective retrieval`.

Y además combina:

`semantic dependencies + document hierarchy`.

### Diferencia

Su problema es **agentic document editing**, no retrieval general.

Por tanto es más una referencia de diseño que un competidor directo.

**Cercanía: 7/10**

---

# 28. Dos trabajos NO incluidos originalmente que yo añadiría

Hay dos líneas que el documento debería haber incluido.

## IRCoT — **9/10**

**Interleaving Retrieval with Chain-of-Thought Reasoning** intercala razonamiento y retrieval para resolver preguntas multi-step. El propio repositorio oficial describe precisamente ese enfoque. ([GitHub][27])

Es relevante porque introduce:

`reasoning → retrieval → new reasoning → retrieval`.

DCS-RAG introduce:

`concept discovery → retrieval → concept discovery → retrieval`.

Son dos formas diferentes de hacer retrieval adaptativo.

---

## Self-RAG — **8/10**

Self-RAG permite que el modelo decida cuándo recuperar, evalúe los resultados y pueda recuperar varias veces durante la generación. ([arXiv][28])

La diferencia es enorme en mecanismo, pero muy importante conceptualmente:

`retrieval no es necesariamente una operación única`.

Eso es exactamente uno de los pilares de DCS-RAG.

---

# 29. También añadiría FLARE — **8/10**

FLARE formula explícitamente el concepto de **active retrieval**: generar una predicción, detectar incertidumbre y recuperar de nuevo de forma iterativa. ([ACL Anthology][29])

Es relevante para la cuestión de:

> "¿cuándo merece la pena volver a recuperar?"

que en DCS-RAG aparece como:

`relevance + novelty + coverage + cost`.

Por tanto FLARE es una referencia muy buena para el **control del ciclo iterativo**, aunque no para el espacio conceptual.

---

# 30. Ranking final de cercanía

Mi ranking, después de revisar la literatura, queda aproximadamente así:

| Posición | Proyecto                                           | Cercanía |
| -------: | -------------------------------------------------- | -------: |
|        1 | **SAE-SPLADE**                                     |  **9,5** |
|        2 | **HyCE-RAG**                                       |  **9,5** |
|        3 | **DyVo**                                           |  **9,0** |
|        4 | **HGRAG**                                          |  **9,0** |
|        5 | **EHRAG**                                          |  **9,0** |
|        6 | **DRIFT Search**                                   |  **9,0** |
|        7 | **LightRAG**                                       |  **9,0** |
|        8 | **PanoramaRAG**                                    |  **9,0** |
|        9 | **SPLADE / SPLADE++**                              |  **8,5** |
|       10 | **GraphRAG**                                       |  **8,5** |
|       11 | **HyperGraphRAG**                                  |  **8,5** |
|       12 | **Query Expansion**                                |  **8,5** |
|       13 | **RAPTOR**                                         |  **8,0** |
|       14 | **Top2Vec**                                        |  **8,0** |
|       15 | **BERTopic**                                       |  **8,0** |
|       16 | **GRAG**                                           |  **8,0** |
|       17 | **FLARE**                                          |  **8,0** |
|       18 | **Self-RAG**                                       |  **8,0** |
|       19 | **ETM**                                            |  **7,5** |
|       20 | **Hyper-RAG**                                      |  **7,5** |
|       21 | **Query2doc**                                      |  **7,0** |
|       22 | **Concept Bottleneck without predefined concepts** |  **7,0** |
|       23 | **LEDGER**                                         |  **7,0** |
|       24 | **LSI**                                            |  **7,0** |
|       25 | **HyDE**                                           |  **6,5** |
|       26 | **DeepCT**                                         |  **6,5** |
|       27 | **Sparse Autoencoders**                            |  **6,5** |

Estas notas son **cercanía arquitectónica conceptual**, no métricas experimentales.

---

# 31. Pero aquí está la conclusión realmente importante

La investigación cambia bastante la interpretación de la propuesta.

## La parte A ya existe

Esto:

> "representar retrieval sparse mediante conceptos en lugar de tokens"

ya existe de forma explícita en **SAE-SPLADE**. ([arXiv][8])

Y esto:

> "usar vocabulario dinámico de entidades/conceptos"

ya existe en **DyVo**. ([arXiv][7])

Y esto:

> "descubrir dimensiones semánticas a partir del corpus"

ya existe en distintas formas en **LSI, ETM, Top2Vec y BERTopic**. ([asistdl.onlinelibrary.wiley.com][1])

Por tanto **no deberíamos presentar la invención como "un espacio conceptual para retrieval"**.

Sería demasiado fuerte y probablemente incorrecto.

---

# 32. La parte B también existe

Esto:

`query → expandir → retrieve → volver a expandir`

tampoco es nuevo.

Existe históricamente como Query Expansion y aparece modernizado en:

* Query2doc;
* DRIFT;
* IRCoT;
* FLARE;
* Self-RAG;
* HyCE-RAG.

([ScienceDirect][12])

---

# 33. La combinación sigue siendo interesante

Donde sí veo algo potencialmente más interesante es en esta combinación concreta:

```text
CORPUS
   │
   ├── Concept Discovery
   │
   ├── Dynamic Concept Dictionary
   │
   ├── Concept Embeddings
   │
   └── Document Hierarchy
          │
          ▼
      CHUNK × CONCEPT
          │
          ▼
   CONCEPT-SPACE RETRIEVAL
          │
          ▼
       QUERY
          │
          ▼
   QUERY → CONCEPTS
          │
          ▼
     INITIAL CHUNKS
          │
          ▼
   DISCOVERED CONCEPTS
          │
          ▼
   QUERY-AWARE EXPANSION
          │
          ▼
      NEW CHUNKS
          │
          ▼
      NEW CONCEPTS
          │
          ▼
       ITERATE
          │
          ▼
   RELEVANCE / NOVELTY /
   COVERAGE / COST
          │
          ▼
    STOP CONDITION
```

**No he encontrado en esta investigación una referencia que reúna exactamente esas piezas de esa manera.**

Pero hay que ser muy cuidadoso: esto **no permite afirmar todavía que nadie lo haya hecho**. Para eso habría que hacer una búsqueda de prior art mucho más sistemática, incluyendo combinaciones de términos, workshops, papers recientes, repositorios y posiblemente patentes.

---

# 34. La diferencia más interesante frente a GraphRAG

Creo que el documento acierta especialmente aquí.

GraphRAG plantea esencialmente:

```text
TEXT
 ↓
ENTITIES
 ↓
RELATIONS
 ↓
GRAPH
 ↓
TRAVERSAL
```

La propuesta plantea:

```text
TEXT
 ↓
CONCEPTS
 ↓
ACTIVATIONS
 ↓
CHUNK × CONCEPT
 ↓
EMERGENT ASSOCIATIONS
 ↓
CONCEPT-SPACE TRAVERSAL
```

La segunda arquitectura **no necesita decidir inmediatamente que "A causa B", "A usa B", "A es parte de B", etc.**

Eso puede ser una ventaja enorme en corpus documentales donde existen muchas relaciones semánticas pero resulta caro o artificial imponer una ontología relacional completa. El propio documento formula esta diferencia explícitamente. 

---

# 35. Y veo una conexión todavía más interesante

La matriz:

```text
             CONCEPTS
        C1   C2   C3   C4   C5
CH1    .9   .8   0    .7   .1
CH2    .8   .1   .6   .7   0
CH3    0    .1   .9   .2   .8
```

puede verse simultáneamente como:

```text
          chunk
            │
            │
       incidence matrix
            │
            ▼
        concepts
```

y de ahí puedes derivar:

```text
chunk ↔ chunk
concept ↔ concept
chunk ↔ concept
```

La literatura más reciente —especialmente HGRAG y EHRAG— está explorando precisamente cómo explotar **estructuras de incidencia y difusión** en lugar de limitarse a embeddings independientes. ([ojs.aaai.org][22])

Esto hace que la matriz `chunk × concept` sea posiblemente **más importante que el propio término "concept embedding"**.

---

# 36. Mi valoración de la novedad

La descompondría así:

| Componente                                           |     Novedad estimada |
| ---------------------------------------------------- | -------------------: |
| Concept discovery                                    |                 2/10 |
| Corpus-specific semantic dimensions                  |                 4/10 |
| Sparse interpretable representation                  |                 3/10 |
| Concept embeddings                                   |                 3/10 |
| Chunk × concept matrix                               |                 5/10 |
| Query → concept mapping                              |                 5/10 |
| Concept-based retrieval                              |                 4/10 |
| Iterative retrieval                                  |                 3/10 |
| Query-aware expansion                                |                 4/10 |
| Concept expansion driven by retrieved chunks         |             **7/10** |
| Dynamic concept-space traversal                      |             **7/10** |
| Concept space + hierarchy                            |             **7/10** |
| Concept space + dense retrieval + structural signals |             **7/10** |
| **Combinación completa**                             | **8/10 provisional** |

La última cifra **no es una afirmación de prioridad ni de patentabilidad**. Es mi evaluación de cuánto espacio conceptual parece quedar después de descontar lo que ya está publicado.

---

# 37. Lo que yo cambiaría en la hipótesis de investigación

En este punto **no investigaría**:

> "¿Podemos crear un espacio conceptual para retrieval?"

Eso ya está suficientemente demostrado como línea de investigación. SAE-SPLADE es la prueba más clara. ([arXiv][8])

Investigaría algo mucho más concreto:

> **¿Puede un espacio conceptual específico del corpus utilizarse como memoria navegable para descubrir, durante el retrieval, conocimiento indirectamente relevante que una query inicial y un retrieval one-shot no recuperarían?**

Y todavía mejor:

> **¿La expansión iterativa sobre un espacio de conceptos inducido por el corpus mejora el coverage del conocimiento necesario por unidad de contexto procesado frente a dense RAG, sparse RAG y GraphRAG?**

Eso sí me parece una pregunta de investigación realmente interesante.

---

# 38. Los tres experimentos que ahora tienen más sentido

Yo haría una comparación mucho más dura que la del documento:

```text
A — Dense RAG
    query → dense retrieval

B — Sparse semantic RAG
    query → SPLADE/SAE-SPLADE → retrieval

C — Concept RAG
    query → concept mapping → chunk × concept retrieval

D — Graph RAG
    query → graph traversal

E — Concept Iterative RAG
    query
      → concept mapping
      → retrieval
      → concept discovery
      → expansion
      → retrieval
      → ...
```

Y mediría especialmente:

```text
Context Recall
Knowledge Coverage
Multi-hop Recall
Answer Completeness
Context Precision
Tokens / answer
Retrieval latency
# iterations
# concepts activated
Novel information / iteration
```

Porque la propuesta no pretende ganar solamente en:

`"¿encontré el documento?"`

sino en:

`"¿descubrí todo lo que necesitaba para contestar?"`

Esa formulación del documento es, en mi opinión, la parte más potente de la hipótesis. 

---

## Conclusión

Mi lectura después de contrastarlo con el estado del arte es:

**No hay una novedad fuerte en ninguna de las piezas individuales.**

Pero sí hay una posible contribución interesante en tratar el **espacio conceptual como un espacio navegable durante retrieval**, y no solamente como una técnica para representar documentos.

La línea que yo consideraría central pasa a ser:

```text
SAE-SPLADE
     +
DyVo
     +
Top2Vec / BERTopic
     +
DRIFT / IRCoT / Self-RAG
     +
HGRAG / EHRAG / HyCE-RAG
     +
RAPTOR
```

pero con una operación distinta:

```text
QUERY
  ↓
CONCEPT SPACE
  ↓
CHUNKS
  ↓
NEW CONCEPTS
  ↓
CONCEPT SPACE
  ↓
NEW CHUNKS
  ↓
...
```

Es decir, **no intentaría vender esto como otro "GraphRAG" ni como otro "sparse retriever"**. El núcleo defendible es mucho más específico: **concept-space traversal para contextual recall**, con la matriz `chunk × concept` como infraestructura de navegación y con retrieval iterativo condicionado por la propia consulta. Esa es la parte que merece ahora una búsqueda de prior art mucho más exhaustiva.

[1]: https://asistdl.onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-4571%28199009%2941%3A6%3C391%3A%3AAID-ASI1%3E3.0.CO%3B2-9?utm_source=chatgpt.com "Indexing by latent semantic analysis - Deerwester - 1990 - Journal of the American Society for Information Science - Wiley Online Library"
[2]: https://aclanthology.org/2020.tacl-1.29/?utm_source=chatgpt.com "Topic Modeling in Embedding Spaces - ACL Anthology"
[3]: https://arxiv.org/abs/2008.09470?utm_source=chatgpt.com "Top2Vec: Distributed Representations of Topics"
[4]: https://arxiv.org/abs/2203.05794?utm_source=chatgpt.com "BERTopic: Neural topic modeling with a class-based TF-IDF procedure"
[5]: https://arxiv.org/abs/1910.10687?utm_source=chatgpt.com "Context-Aware Sentence/Passage Term Importance Estimation For First Stage Retrieval"
[6]: https://arxiv.org/abs/2107.05720?utm_source=chatgpt.com "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking"
[7]: https://arxiv.org/abs/2410.07722?utm_source=chatgpt.com "DyVo: Dynamic Vocabularies for Learned Sparse Retrieval with Entities"
[8]: https://arxiv.org/abs/2604.21511?utm_source=chatgpt.com "From Tokens to Concepts: Leveraging SAE for SPLADE"
[9]: https://arxiv.org/abs/2309.08600?utm_source=chatgpt.com "Sparse Autoencoders Find Highly Interpretable Features in Language Models"
[10]: https://transformer-circuits.pub/2023/monosemantic-features/index.html?utm_source=chatgpt.com "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"
[11]: https://arxiv.org/abs/2407.03921?utm_source=chatgpt.com "Concept Bottleneck Models Without Predefined Concepts"
[12]: https://www.sciencedirect.com/science/article/pii/S0306457318305466?utm_source=chatgpt.com "Query expansion techniques for information retrieval: A survey - ScienceDirect"
[13]: https://aclanthology.org/2023.emnlp-main.585/?utm_source=chatgpt.com "Query2doc: Query Expansion with Large Language Models - ACL Anthology"
[14]: https://aclanthology.org/2023.acl-long.99/?utm_source=chatgpt.com "Precise Zero-Shot Dense Retrieval without Relevance Labels - ACL Anthology"
[15]: https://www.microsoft.com/en-us/research/project/graphrag/overview/?utm_source=chatgpt.com "Project GraphRAG - Microsoft Research: Overview"
[16]: https://microsoft.github.io/graphrag/query/drift_search/?utm_source=chatgpt.com "DRIFT Search - GraphRAG"
[17]: https://arxiv.org/abs/2410.05779?utm_source=chatgpt.com "LightRAG: Simple and Fast Retrieval-Augmented Generation"
[18]: https://arxiv.org/abs/2401.18059?utm_source=chatgpt.com "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
[19]: https://arxiv.org/abs/2405.16506?utm_source=chatgpt.com "GRAG: Graph Retrieval-Augmented Generation"
[20]: https://proceedings.neurips.cc/paper_files/paper/2025/hash/df55ee6e59f8ac4a625219e11fe9ddba-Abstract-Conference.html?utm_source=chatgpt.com "HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation"
[21]: https://doi.org/10.1038/s41467-026-71411-1?utm_source=chatgpt.com "Hyper-RAG: combating LLM hallucinations using hypergraph-driven retrieval-augmented generation | Nature Communications"
[22]: https://ojs.aaai.org/index.php/AAAI/article/view/40623?utm_source=chatgpt.com "Cross-Granularity Hypergraph Retrieval-Augmented Generation for Multi-hop Question Answering | Proceedings of the AAAI Conference on Artificial Intelligence"
[23]: https://aclanthology.org/2026.findings-acl.1233/?utm_source=chatgpt.com "EHRAG: Bridging Semantic Gaps in Lightweight GraphRAG via Hybrid Hypergraph Construction and Retrieval - ACL Anthology"
[24]: https://arxiv.org/abs/2607.22597?utm_source=chatgpt.com "HyCE-RAG: Hypergraph Chain-of-Evidence Retrieval-Augmented Generation for Explainable Multi-hop Question Answering"
[25]: https://aclanthology.org/2026.findings-acl.1998/?utm_source=chatgpt.com "PanoramaRAG: Enabling Consistent Global Topic Awareness in Graph-Based RAG - ACL Anthology"
[26]: https://aclanthology.org/2026.findings-acl.515/?utm_source=chatgpt.com "LEDGER: Scaling Agentic Document Editing with Dependency-aware Graph Retrieval - ACL Anthology"
[27]: https://github.com/StonyBrookNLP/ircot?utm_source=chatgpt.com "GitHub - StonyBrookNLP/ircot: Repository for Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions, ACL23 · GitHub"
[28]: https://arxiv.org/abs/2310.11511?utm_source=chatgpt.com "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
[29]: https://aclanthology.org/2023.emnlp-main.495/?utm_source=chatgpt.com "Active Retrieval Augmented Generation - ACL Anthology"
