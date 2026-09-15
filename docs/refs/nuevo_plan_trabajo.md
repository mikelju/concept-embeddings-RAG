Sí. Y creo que aquí merece la pena ser **muy disciplinado**: no tocaría el código de la Fase 4 para intentar “arreglarlo”, sino que la cerraría como un resultado experimental válido y abriría una nueva fase que vuelva a probar la hipótesis en la representación que realmente queríamos estudiar.

He revisado además la estructura real de `concept-embeddings-RAG`: en la rama actual existen `phase_4/4.spec.md`, `4.0_query_aware_iterative_expansion.md` y `4.tasks.md`, y el `master_plan` todavía considera pendientes las fases posteriores. 

## La dirección que propongo

No haría:

```text
Fase 4
   ↓
tuning
   ↓
más restart / más K / más alpha
```

Haría:

```text
Fase 4
   ↓
CIERRE FORMAL
   ↓
resultado: pooled-embedding concepts = negativo
   ↓
merge a main
   ↓
NUEVA FASE 5
   ↓
conceptos extraídos del texto
   ↓
piloto muy pequeño sobre 50 preguntas
   ↓
¿la navegación mejora?
   ↓
sí ---------------- no
↓                     ↓
Fase 6 completa       cierre de hipótesis
```

El motivo es que el experimento actual ya ha cubierto bastante bien el espacio de variantes de **ese** diseño: K, operadores de query, vistas, damping, fusión y difusión.  

---

# PLAN DE TRABAJO PARA EL EQUIPO

## BLOQUE 0 — Cerrar correctamente la investigación que ya existe

### Paso 1 — Congelar la rama actual

En la rama:

```text
phase-4-iterative-expansion
```

hacer una revisión final del estado de trabajo:

```bash
git status
git log --oneline --decorate -20
pytest
```

Comprobar que:

* no quedan cambios funcionales sin commitear;
* los resultados de Fase 4 corresponden al commit declarado;
* todos los artefactos necesarios están presentes;
* `pytest` sigue pasando.

**No modificar el algoritmo de Fase 4.**

El objetivo es conservar exactamente la implementación cuyo resultado estamos evaluando.

---

### Paso 2 — Crear un commit explícito de cierre de Fase 4

Haría un commit de este estilo:

```text
docs: close Phase 4 with negative result
```

o, si todavía hay algún pequeño cambio puramente documental:

```text
docs: document Phase 4 findings and experimental limitations
```

El commit debe dejar claro que **el código experimental está cerrado** y que cualquier nueva estrategia será otra fase.

---

### Paso 3 — Documentar el resultado en `phase_4/4.results.md`

Si todavía no existe un `4.results.md` formal, crearlo.

Debe contener cuatro bloques muy concretos:

#### 3.1 Resultado

```text
System C does not improve retrieval over dense retrieval.
```

#### 3.2 Resultado mecánico

Documentar:

* no se promocionan nuevos párrafos;
* la difusión alcanza casi todo el pool;
* la semilla domina el ranking;
* la expansión se convierte en un reordenamiento del top-100. 

#### 3.3 Qué queda refutado

Escribir explícitamente:

> The experiment rejects the current diffusion formulation over the pooled-embedding-derived concept space for this benchmark.

No escribir:

> Concept-based retrieval is disproved.

#### 3.4 Qué queda abierto

Incluir literalmente estas tres cuestiones:

1. conceptos extraídos del texto;
2. representación más fina que pooled embeddings;
3. benchmark diseñado para contexto satélite / temática.

Es justamente lo que el informe identifica como no probado. 

---

# BLOQUE 1 — Registrar el aprendizaje en el Master Plan

### Paso 4 — Modificar `docs/plans/0_master_plan.md`

Aquí haría un cambio importante.

La frase actual de la Fase 4 dice que su misión es “navegar” el espacio conceptual y todavía deja la fase como parte del camino abierto.

Añadiría inmediatamente después de la Fase 4 una sección:

```markdown
### Phase 4 conclusion — negative result, limited scope

Phase 4 establishes that diffusion over the concept space induced by pooled
document embeddings does not provide useful retrieval expansion on HotpotQA.

The failure is attributable to the current representation and diffusion
formulation: the learned atoms behave primarily as broad semantic/entity-type
directions, propagation reaches nearly the whole corpus, and no unseen seed
unit is promoted into the final top-100.

This result does NOT test the original architecture in which concepts are
extracted from document text. That architecture remains untested.
```

Esto es crucial.

Así el repositorio queda históricamente limpio:

```text
original hypothesis
       ↓
implementation A
       ↓
negative result A
       ↓
new experiment
```

y no parece que hayamos cambiado retrospectivamente la hipótesis.

---

# BLOQUE 2 — Cambiar la estructura de fases

Aquí haría el cambio más importante de planificación.

## Paso 5 — No usar la actual Fase 5 como “comparative evaluation final”

La actual Fase 5 dice:

> Full benchmark + ablations + MuSiQue + final verdict.

Pero **todavía no estamos preparados para ese cierre**, porque acabamos de descubrir que la operacionalización del concepto es la variable que necesitamos cambiar. El propio master plan deja abierta esa fase.

Yo reorganizaría:

### Fase 5

**Text-derived concept representation and navigation pilot**

### Fase 6

**Comparative evaluation and verdict**

### Fase 7

**Exploratory extensions**

Es decir:

```text
Phase 1  baseline
Phase 2  pooled-embedding concept space      ← probado
Phase 3  conceptual retrieval                ← probado
Phase 4  diffusion                           ← probado

Phase 5  TEXT-DERIVED CONCEPTS                ← NUEVA
Phase 6  FINAL COMPARATIVE EVALUATION
Phase 7  OPTIONAL EXTENSIONS
```

No necesitamos borrar nada de las fases anteriores.

---

# BLOQUE 3 — Crear formalmente la nueva Fase 5

### Paso 6 — Crear:

```text
docs/plans/phase_5/5.spec.md
docs/plans/phase_5/5.0_text_derived_concepts.md
docs/plans/phase_5/5.tasks.md
```

Y seguir el flujo que ya utiliza el proyecto:

```text
spec
→ plan
→ tasks
→ implementation
→ verification
→ audit
→ results
```

Esto encaja con la propia convención documental del repositorio.

---

# BLOQUE 4 — Definir exactamente qué queremos probar

## Paso 7 — Especificar la nueva hipótesis

La nueva hipótesis experimental debe ser muy concreta:

> **Do explicit concepts and entities extracted from document text provide a more useful navigational representation than concepts induced from pooled document embeddings?**

Y una segunda:

> **Can navigation over those text-derived nodes recover relevant unseen documents from an initial dense retrieval more effectively than continuing the dense ranking or using BM25?**

Esta formulación es mejor que “¿funciona Concept Embeddings RAG?”.

Porque ahora estamos aislando la variable correcta.

---

# BLOQUE 5 — Elegir las 50 preguntas

Este paso lo considero **muy importante**.

## Paso 8 — Construir el conjunto de 50 preguntas

No debemos elegirlas manualmente.

Eso introduciría sesgo.

El conjunto debe salir de las 72 preguntas donde ambos baselines fallan.

Pero además aplicaría un filtro relacionado con el mecanismo que queremos probar:

```text
dense top-10
       ↓
gold paragraph missing
       ↓
bridge question
       ↓
title/entity of gold appears
inside one of the already retrieved paragraphs
```

Esto es exactamente el escenario en el que esperamos que una representación explícita pueda navegar.

Ya sabemos que aproximadamente el 68,4 % de los párrafos faltantes tienen el título de la entidad dentro de los diez párrafos ya recuperados. 

Hay suficiente población para sacar 50 sin problemas.

### Regla de selección

Los 50 deben ser:

* seleccionados automáticamente;
* mediante una semilla fija;
* antes de ejecutar la nueva extracción;
* sin mirar cuáles tienen más probabilidad de ganar.

Por ejemplo:

```text
candidate set = all dev questions satisfying the frozen criteria
sort by hash(seed + question_id)
take first 50
```

Guardar inmediatamente:

```text
data/phase5/pilot_questions.json
```

con:

```json
{
  "seed": 42,
  "selection_rule": "...",
  "question_ids": [...]
}
```

Y congelarlo.

---

# BLOQUE 6 — No extraer únicamente “conceptos”

Aquí cambiaría ligeramente mi recomendación anterior.

## Paso 9 — Extraer tres tipos de información

Para cada párrafo:

### A. Entities

Ejemplo:

```text
Kaley Cuoco
Sean M. Carroll
The Big Bang Theory
```

### B. Concepts / topics

```text
television actor
sitcom
science communication
```

### C. Key relations / attributes, si son fácilmente identificables

```text
hosted
born in
editor of
played for
```

Pero **no construir todavía un knowledge graph formal**.

Queremos probar el mínimo mecanismo necesario.

---

# BLOQUE 7 — Diseñar la representación

## Paso 10 — Crear un nuevo espacio:

$$
X_{text} \in \mathbb{R}^{chunks \times nodes}
$$

donde:

```text
nodes = entities + concepts (+ optional attributes)
```

Ejemplo:

| Chunk | Kaley Cuoco | actor | sitcom | host |  TV |
| ----- | ----------: | ----: | -----: | ---: | --: |
| P1    |         1.0 |   0.7 |    0.9 |  0.2 | 0.8 |

Al principio **no necesitamos pesos sofisticados**.

Podemos empezar con:

```text
entity present = 1
concept present = 1
relation present = 1
```

y, en una segunda iteración, añadir pesos.

Esto es deliberado: queremos probar primero la estructura, no el scoring.

---

# BLOQUE 8 — No hacer 19.366 llamadas de LLM todavía

Este es probablemente el cambio de proceso más importante respecto a cómo hemos trabajado antes.

## Paso 11 — Ejecutar sólo el piloto de 50 preguntas

No procesar todavía todo el corpus.

Extraer conceptos/entidades solamente de:

* los top-10 densos de las 50 preguntas;
* los párrafos gold;
* y, si es necesario para buscar candidatos, el subconjunto relevante del pool.

Así podremos medir:

```text
¿funciona la idea?
```

antes de gastar dinero y tiempo construyendo toda la estructura.

El propio informe propone un piloto similar precisamente por esta razón. 

---

# BLOQUE 9 — Mantener el mismo baseline

## Paso 12 — No cambiar simultáneamente el retriever inicial

Para esta primera prueba:

```text
seed = dense
```

No cambiar:

* modelo embedding;
* corpus;
* chunking;
* top-10;
* preguntas;
* presupuesto.

Queremos aislar:

```text
old representation
vs
new representation
```

No:

```text
old everything
vs
new everything
```

---

# BLOQUE 10 — Probar el salto de forma extremadamente simple

## Paso 13 — Implementar primero un “one-hop navigation baseline”

Antes de volver al random walk completo:

```text
dense top-10
      ↓
nodes discovered in top-10
      ↓
retrieve chunks sharing those nodes
```

Es decir:

$$
score(v)=\sum_{n \in Nodes(p_1)} w_n \cdot X[v,n]
$$

Esto reproduce exactamente la pregunta del diagnóstico que ya hicisteis, pero usando nodos extraídos del texto.

---

# BLOQUE 11 — Comparación mínima

## Paso 14 — Comparar estas seis estrategias

Para las 50:

| Estrategia             | Qué prueba                  |
| ---------------------- | --------------------------- |
| Dense continuation     | baseline                    |
| BM25(query)            | lexical baseline            |
| BM25(texto p1)         | seguir entidades explícitas |
| Dense neighbours(p1)   | similitud semántica         |
| Old concept space      | método actual               |
| **Text-derived nodes** | **nueva hipótesis**         |

El objetivo principal será:

$$
Recall@10
$$

sobre el párrafo gold que falta.

Y secundarias:

$$
Recall@100
$$

$$
new\ candidates
$$

$$
candidate\ concentration
$$

---

# BLOQUE 12 — Añadir una métrica muy importante

## Paso 15 — Medir “expansion gain”

Quiero que añadáis explícitamente:

$$
ExpansionGain =
Recall_{expanded}
-
Recall_{seed}
$$

Pero además:

$$
NewRelevantHit =
\text{gold found by expansion and absent from seed}
$$

La pregunta clave ya no es únicamente:

> “¿qué sistema tiene mejor recall?”

sino:

> **“¿cuántas veces consigue la estructura conceptual traer algo que el retriever inicial no tenía?”**

Eso es mucho más directamente fiel a la hipótesis.

---

# BLOQUE 13 — Analizar de dónde proceden las victorias

## Paso 16 — Para cada éxito registrar el camino

Para cada pregunta recuperada gracias a la nueva estructura:

```text
query
 ↓
dense paragraph P1
 ↓
entity/concept X
 ↓
candidate paragraph P2
 ↓
gold
```

Y guardar:

```json
{
  "question_id": "...",
  "seed_unit": "...",
  "bridge_node": "...",
  "new_unit": "...",
  "gold": true
}
```

Esto nos permitirá saber si realmente estamos viendo:

```text
semantic navigation
```

o simplemente:

```text
normalized lexical matching
```

---

# BLOQUE 14 — Decisión tras el piloto

## Paso 17 — Establecer un gate antes de escalar

No lanzaría inmediatamente todo el corpus.

Definiría un criterio de continuación.

Por ejemplo:

### GO

Si text-derived nodes:

* recuperan significativamente más golds nuevos que old concepts;
* superan claramente dense continuation;
* y lo hacen mediante rutas explicables.

Entonces:

```text
→ escalar al corpus completo
→ implementar X completo
→ implementar iterative expansion
```

### STOP

Si:

```text
text-derived concepts ≤ dense continuation
```

y no producen expansión útil:

```text
→ no invertir más en Concept Embeddings RAG
→ pasar a cierre del proyecto
```

Esto evita volver a caer en seis fases de desarrollo antes de saber si la hipótesis tiene señal.

---

# BLOQUE 15 — Sólo después construir el corpus completo

## Paso 18 — Si el piloto da GO

Entonces sí:

```text
19,366 chunks
        ↓
LLM extraction
        ↓
entities + concepts
        ↓
normalization
        ↓
X_text
```

Y aquí sí tiene sentido estudiar:

* frecuencia;
* pesos;
* deduplicación;
* sinonimia;
* hierarquía;
* embeddings de conceptos;
* sparsity.

---

# BLOQUE 16 — Sólo después recuperar la expansión iterativa

## Paso 19 — Recuperar la idea de difusión, pero sobre nodos mejores

La nueva arquitectura sería:

```text
query
 ↓
dense seed
 ↓
entities/concepts from retrieved chunks
 ↓
node scoring
 ↓
new chunks
 ↓
new nodes
 ↓
repeat
```

Y sólo entonces introduciríamos:

$$
s_{t+1}
=
r\,s_0 + (1-r)\,P(s_t)
$$

porque ahora **P** estaría operando sobre una estructura donde un nodo puede ser:

```text
Kaley Cuoco
```

en vez de:

```text
Wikipedia-style entity overview
```

---

# BLOQUE 17 — Corregir la filosofía de la difusión

Aquí haría otra modificación importante respecto a Fase 4.

No volvería a usar:

```text
seed top-100
+ diffusion
+ restart >= 0.2
```

como única política.

La nueva exploración debería permitir:

```text
exploitation: seed
+
exploration: new nodes
```

y reservar explícitamente parte del presupuesto para nuevos candidatos.

Por ejemplo:

```text
80% seed / 20% expansion
```

o:

```text
top 8 seed
+ top 8 expanded
```

Esto **no tiene que ser el diseño definitivo**. Primero necesitamos comprobar que los nodos son útiles.

---

# BLOQUE 18 — Mover el antiguo “comparative evaluation” a Fase 6

## Paso 20 — Reescribir la antigua Fase 5 como Fase 6

Cuando la Fase 5 termine, la Fase 6 deberá comparar:

```text
BM25
Dense
Dense + BM25
Old Concept Space
Text-derived Concept Space
Iterative Text-derived Concept Space
```

Y entonces sí:

* 512 / 1024 / 2048 / 4096 tokens;
* Full Support;
* Gold Recall;
* Context Precision;
* coste;
* latencia;
* iteraciones.

---

# BLOQUE 19 — MuSiQue después, no antes

## Paso 21 — Dejar MuSiQue para la Fase 6

No utilizaría MuSiQue ahora.

Primero:

```text
¿funciona el mecanismo?
```

Después:

```text
¿generaliza?
```

Así evitamos gastar muchísimo esfuerzo en un sistema que quizá todavía no tiene señal.

---

# BLOQUE 20 — Qué hacer con la Fase 7

## Paso 22 — Dejar como opcional

La actual Fase 6 de extensiones pasa a Fase 7:

```text
document hierarchy
variable resolution
graph derived from X
SAE/token-level representation
```

Y sólo se abre si la Fase 6 lo justifica.

---

# La estructura final que yo dejaría en GitHub

Quedaría así:

```text
docs/
└── plans/
    ├── 0_master_plan.md
    │
    ├── phase_1/
    ├── phase_2/
    ├── phase_3/
    ├── phase_4/
    │   ├── 4.spec.md
    │   ├── 4.0_query_aware_iterative_expansion.md
    │   ├── 4.tasks.md
    │   └── 4.results.md              ← cierre negativo
    │
    ├── phase_5/
    │   ├── 5.spec.md                 ← nueva hipótesis
    │   ├── 5.0_text_derived_concepts.md
    │   ├── 5.tasks.md
    │   └── 5.results.md
    │
    ├── phase_6/
    │   ├── 6.spec.md
    │   ├── 6.0_comparative_evaluation.md
    │   └── 6.results.md
    │
    └── phase_7/
```

Y los artefactos nuevos:

```text
data/
└── phase5/
    ├── pilot_questions.json
    ├── dense_seed.json
    ├── extracted_nodes/
    ├── node_matrix/
    ├── one_hop_results.json
    └── traces/
```

---

# Orden exacto de trabajo que yo daría al técnico

Te lo dejaría literalmente así, para que puedas pasárselo:

### Sprint 1 — Cierre

**1.** Congelar `phase-4-iterative-expansion`.

**2.** Ejecutar `pytest`.

**3.** Comprobar que resultados y artefactos de Fase 4 reproducen el commit declarado.

**4.** Crear `docs/plans/phase_4/4.results.md`.

**5.** Documentar como conclusión:
`pooled-embedding-derived concept space + current diffusion = negative result`.

**6.** Documentar explícitamente que esto **no refuta text-derived concepts**.

**7.** Actualizar `docs/plans/0_master_plan.md`.

**8.** Marcar Fase 4 como `Complete — Negative Result`.

**9.** Commit.

**10.** Abrir PR `phase-4-iterative-expansion → main`.

**11.** Revisar y mergear.

---

### Sprint 2 — Preparar la nueva investigación

**12.** Crear `phase_5`.

**13.** Escribir `5.spec.md`.

**14.** Definir formalmente que ahora el concepto procede del **texto**, no del pooled embedding.

**15.** Definir entidades + conceptos como nodos de primera versión.

**16.** Definir que el LLM se usa **offline para construir la representación**, no dentro del loop de retrieval.

**17.** Crear la regla determinista de selección del conjunto piloto.

**18.** Generar las 50 preguntas desde el conjunto elegible de dev.

**19.** Guardar y congelar `pilot_questions.json`.

---

### Sprint 3 — Piloto

**20.** Recuperar top-10 dense para esas 50 preguntas.

**21.** Extraer entidades y conceptos de esos documentos.

**22.** Extraer entidades/conceptos del gold.

**23.** Construir el primer `X_text` del piloto.

**24.** Implementar one-hop text-derived navigation.

**25.** Compararlo con dense continuation, BM25 y old concept space.

**26.** Registrar todos los caminos de expansión.

**27.** Calcular `Recall@10`, `Recall@100`, `ExpansionGain` y número de golds nuevos.

**28.** Generar `5.results.md`.

---

### Sprint 4 — Gate

**29.** Aplicar el criterio GO/STOP.

**30.** Si STOP → documentar resultado y cerrar la línea.

**31.** Si GO → ampliar al corpus completo.

**32.** Sólo entonces implementar iterative expansion.

**33.** Sólo entonces preparar Fase 6.

---

# Y hay una cosa que cambiaría respecto al plan anterior

**No intentaría ahora convertir los conceptos extraídos por LLM directamente en embeddings y hacer nearest-neighbor como mecanismo principal.**

Primero probaría:

```text
query
 ↓
retrieve seed
 ↓
extract explicit nodes from seed
 ↓
navigate via node incidence
```

Porque la pregunta que queremos contestar primero es:

> **¿la estructura explícita del contenido permite saltar a conocimiento que dense no recuperó?**

Después ya podemos usar embeddings para hacer el mapping semántico:

```text
"adaptar el modelo"
       ↓
fine-tuning
model adaptation
instruction tuning
```

Eso será la segunda capa, no la primera.

---

## En una frase: el giro que estamos haciendo

El proyecto pasa de intentar demostrar:

> **“podemos convertir embeddings en conceptos y navegar esos conceptos”**

a probar:

> **“podemos convertir el contenido del corpus en una representación explícita de entidades y conceptos y usarla como una memoria navegable que descubre información que el retrieval one-shot no encuentra”.**

Y esto, para mí, **sí está mucho más cerca de la idea que originalmente concebimos**. La Fase 4 ha sido útil precisamente porque nos ha demostrado qué no hacer.

La arquitectura actual ya deja claro que `X` es la pieza central y que la implementación de retrieval conceptual no accede a etiquetas ni texto, sino exclusivamente a esa representación inducida desde embeddings.  La nueva Fase 5 debe romper deliberadamente esa restricción **en la etapa de construcción de X**, que es exactamente donde está el cambio científico que necesitamos.
