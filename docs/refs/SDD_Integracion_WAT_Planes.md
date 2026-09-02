# Integración SDD en el Framework WAT + Planes

## Guía práctica para proyectos de 30–100 horas

---

## El diagnóstico: qué tienes, qué falta y dónde encaja SDD

### Lo que ya tienes (y funciona bien)

Tu sistema actual cubre dos de los cuatro pilares del desarrollo con agentes:

**WAT (Workflows → Agents → Tools)** resuelve el problema de la ejecución: separar la inteligencia (el agente) de la acción (los scripts). El agente razona, los tools ejecutan. Esto ya te da fiabilidad en la capa de "hacer cosas".

**El sistema de Planes** resuelve el problema del seguimiento: saber qué se ha planificado, qué se ha hecho, qué ha salido mal y cómo se ha corregido. Con el plan maestro, los planes de fase, las desviaciones y los fixes, tienes un historial completo y trazable de la vida del proyecto.

### Lo que falta

Tu sistema tiene un hueco entre el plan maestro (visión general) y los planes de fase (cómo implementar). Ese hueco es la **especificación funcional**: el documento que define con precisión QUÉ debe hacer cada funcionalidad, con qué restricciones, qué criterios de aceptación y qué está fuera de alcance.

Hoy, tus planes de fase saltan directamente a los pasos de implementación ("Paso 1: crear el modelo de datos", "Paso 2: montar el endpoint"). Esto funciona cuando tú, como desarrollador, tienes el contexto completo en la cabeza. Pero cuando un agente de IA lee un plan de fase sin una especificación funcional previa, se ve obligado a adivinar:

- ¿Qué campos tiene el modelo de datos? → Lo inventa
- ¿Qué pasa si el usuario envía datos incorrectos? → Lo decide él
- ¿Qué formato tiene la respuesta de la API? → Lo asume
- ¿Qué NO debe hacer el sistema? → No lo sabe

**SDD cubre exactamente ese hueco.** Añade la capa de "especificación de intención" entre la visión (plan maestro) y la ejecución (plan de fase).

### Cómo encajan los tres sistemas

```
┌─────────────────────────────────────────────────────┐
│                    TU FRAMEWORK                      │
│                                                      │
│   WAT                 SDD (nuevo)      PLANES        │
│   ─────               ──────────       ──────        │
│   Cómo se ejecuta  +  Qué se pide  +  Cómo se       │
│   el trabajo           al agente       documenta y   │
│   (workflows/tools)    (specs)         rastrea todo   │
│                                        (plans/fixes)  │
│                                                      │
│   "El agente lee      "El agente       "Todo queda   │
│    instrucciones y     sabe qué         registrado:   │
│    llama scripts"      construir        plan, cambios, │
│                        con precisión"   errores"       │
└─────────────────────────────────────────────────────┘
```

**WAT** no cambia. Sigue siendo tu motor de ejecución.
**Planes** no cambian. Siguen siendo tu sistema de seguimiento.
**SDD** se integra como la capa que alimenta a los planes con información precisa y sin ambigüedad.

---

## La estructura de archivos integrada

### Cambio principal: se añade un archivo `spec.md` en cada carpeta de fase

Tu estructura actual de planes no cambia. Solo se añade un archivo nuevo por fase. El archivo `spec.md` se escribe ANTES del plan de fase y es el que lo alimenta.

```
docs/plans/
├── 0_plan_maestro.md              # Sin cambios — visión global y fases
├── fase_1/
│   ├── 1.spec.md                  # ← NUEVO: Especificación funcional (QUÉ + POR QUÉ)
│   ├── 1.0_nombre_fase.md        # Sin cambios — Plan de implementación (CÓMO)
│   ├── 1.1_desviacion.md         # Sin cambios — Desviaciones
│   └── 1.2_ajuste.md             # Sin cambios — Ajustes
├── fase_2/
│   ├── 2.spec.md                  # ← NUEVO
│   ├── 2.0_nombre_fase.md
│   └── ...
└── fixes/
    └── fix-N_nombre.md            # Sin cambios — Bug fixes
```

**¿Por qué `X.spec.md` y no `X.0_spec.md`?**

Para no romper tu convención de numeración (`X.0` = plan principal, `X.Y` = desviaciones). El archivo de spec usa una extensión distinta (`.spec.md`) que lo identifica inmediatamente como un documento SDD. El plan de fase (`X.0`) sigue siendo el plan de implementación como hasta ahora.

### El CLAUDE.md absorbe la "constitución" de SDD

SDD habla de un archivo `constitution.md` que define las reglas sagradas del proyecto. Tú ya tienes eso: es tu `CLAUDE.md`. No necesitas crear un archivo nuevo. Solo hace falta enriquecer CLAUDE.md con dos secciones que hoy no incluye:

```markdown
# Secciones a añadir en CLAUDE.md

## Permisos del agente

✅ SIEMPRE (hacer sin preguntar):
- Ejecutar tests antes de dar por completada una tarea
- Seguir las convenciones de nombrado del proyecto
- [adaptar a cada proyecto]

⚠️ PREGUNTAR PRIMERO (suspender y pedir confirmación):
- Modificar esquemas de base de datos
- Añadir nuevas dependencias
- [adaptar a cada proyecto]

🚫 NUNCA (prohibido sin excepción):
- Commitear secretos o credenciales
- Escribir lógica de negocio en routers/controladores
- [adaptar a cada proyecto]

## Anti-patrones del proyecto

🚫 No usar bloques try/catch con log y rethrow — el manejo de errores
   está centralizado en [middleware/mecanismo específico].
🚫 No usar [patrón X] — en este proyecto usamos [patrón Y] porque [razón].
🚫 [otros anti-patrones específicos del proyecto]
```

### Estructura completa del proyecto con las tres capas

```
mi-proyecto/
├── CLAUDE.md                      # Constitución: reglas + arquitectura + permisos
├── docs/
│   ├── plans/                     # Sistema de Planes (sin cambios + specs)
│   │   ├── 0_plan_maestro.md
│   │   ├── fase_1/
│   │   │   ├── 1.spec.md         # SDD: especificación funcional
│   │   │   ├── 1.0_nombre.md     # Plan de implementación
│   │   │   └── 1.Y_*.md          # Desviaciones
│   │   └── fixes/
│   ├── brand/                     # Business Brain (sin cambios)
│   ├── learnings/                 # Self-learning (sin cambios)
│   └── refs/                      # Referencias inmutables (sin cambios)
├── workflows/                     # WAT: instrucciones de proceso
├── tools/                         # WAT: scripts de ejecución
├── src/                           # Código
├── tests/                         # Tests
└── .env                           # Credenciales
```

---

## El proceso completo: de cero a implementación

### Vista general del flujo

```
Paso 0: Visión          →  /crea-plan-maestro    →  0_plan_maestro.md
Paso 1: Scaffold        →  crear proyecto base   →  package.json, etc.
Paso 2: Constitución    →  /init-project (mejorado) →  CLAUDE.md (con permisos + anti-patrones)
Paso 3: Especificar     →  /especificar (NUEVO)  →  X.spec.md (por cada fase)
Paso 4: Planificar      →  /planear (mejorado)   →  X.0_nombre.md (lee la spec)
Paso 5: Implementar     →  trabajo normal        →  código + tests
Paso 6: Verificar       →  tests + revisión      →  marcar tareas [x]
         ↓
      (repetir pasos 3-6 para cada fase)
```

Los pasos 0, 1 y 2 se ejecutan una vez al inicio del proyecto.
Los pasos 3 a 6 se repiten por cada fase.

### Paso 0: Visión (`/crea-plan-maestro`)

**Sin cambios.** Funciona exactamente como ahora. Genera `0_plan_maestro.md` con las fases del proyecto.

Resultado: sabes QUÉ fases tiene el proyecto y en qué orden.

### Paso 1: Scaffold del proyecto

**Sin cambios.** Creas la estructura base del código (`npm create vite@latest`, `npx create-next-app`, etc.).

Resultado: hay código real que analizar.

### Paso 2: Constitución (`/init-project` mejorado)

**Cambio menor.** El comando `/init-project` funciona igual que ahora pero, al generar CLAUDE.md, añade automáticamente las secciones de "Permisos del agente" y "Anti-patrones" con valores por defecto que el usuario revisa y adapta.

El protocolo actualizado de `/init-project` añade un paso entre el 4 (identificar gotchas) y el 5 (generar CLAUDE.md):

```
### Paso 4b: Definir permisos y anti-patrones

Pregunta al usuario:
1. "¿Hay acciones que el agente deba hacer SIEMPRE sin preguntar?"
   (ej. ejecutar tests, seguir convenciones de nombrado)
2. "¿Hay acciones donde deba PARAR y pedirte confirmación?"
   (ej. cambiar esquema de BD, añadir dependencias, tocar CI/CD)
3. "¿Hay patrones o prácticas que estén PROHIBIDAS en este proyecto?"
   (ej. try/catch con rethrow, lógica en controladores, etc.)

Si el usuario no tiene preferencias claras, usar estos valores por defecto:

✅ SIEMPRE: ejecutar tests, seguir convenciones de CLAUDE.md, tipar todo
⚠️ PREGUNTAR: cambios de esquema, nuevas dependencias, cambios de infra
🚫 NUNCA: commitear secretos, borrar tests sin resolver, editar archivos generados
```

Resultado: CLAUDE.md contiene todo lo que SDD llama "constitución" — reglas, arquitectura, permisos y prohibiciones.

### Paso 3: Especificar (`/especificar` — NUEVO)

Este es el paso nuevo que aporta SDD. Se ejecuta antes de cada fase de implementación. El comando lee el plan maestro, identifica la fase a especificar, y genera el archivo `X.spec.md` a través de una conversación breve con el usuario.

**Tiempo estimado: 10–20 minutos por fase** (para proyectos de 30–100h).

El protocolo detallado del comando `/especificar` está más abajo.

Resultado: cada fase tiene un documento claro de QUÉ hacer, con criterios de aceptación medibles, contratos de datos, restricciones y anti-objetivos.

### Paso 4: Planificar (`/planear` mejorado)

**Cambio menor pero importante.** El protocolo de `/planear` se modifica para que, al crear un plan de fase, lea primero el `X.spec.md` correspondiente. Esto garantiza que el plan de implementación (el CÓMO) esté anclado a la especificación funcional (el QUÉ).

El cambio concreto en el protocolo de `/planear` es añadir un paso antes del paso 3 actual:

```
### Paso 2b: Leer la especificación funcional

Si existe `docs/plans/fase_X/X.spec.md`:
- Leerlo completo.
- El plan de fase que se cree DEBE cubrir todos los criterios
  de aceptación definidos en la spec.
- Los contratos de datos y API de la spec son INVARIANTES:
  el plan no puede contradecirlos.

Si NO existe:
- Comunicar al usuario: "No hay especificación funcional para esta fase.
  Ejecuta /especificar primero para definir QUÉ hay que construir
  antes de planificar CÓMO."
- No continuar hasta que exista la spec (o el usuario decida
  explícitamente saltársela para esta fase).
```

Resultado: los planes de fase ahora nacen de una especificación precisa en vez de la interpretación del agente.

### Pasos 5 y 6: Implementar y verificar

**Sin cambios.** El trabajo de implementación sigue igual: el agente ejecuta tareas del plan de fase, usa WAT (workflows + tools) cuando corresponde, documenta desviaciones si aparecen, y verifica contra los criterios de aceptación que ahora están explícitos en la spec.

La única diferencia es que ahora existe una referencia objetiva para verificar: los criterios de aceptación de `X.spec.md` se pueden comprobar uno a uno al terminar la fase.

---

## El comando `/especificar`: protocolo completo

```markdown
# Comando /especificar

Genera el archivo de especificación funcional (X.spec.md)
para una fase del plan maestro.
Se ejecuta ANTES de /planear para cada fase nueva.

---

## Cuándo usar este comando

- Antes de empezar una fase nueva que aún no tiene spec.
- Cuando el alcance de una fase ha cambiado y la spec necesita
  actualizarse.

---

## Protocolo

### Paso 1: Cargar contexto

Lee en este orden:
1. CLAUDE.md (constitución del proyecto)
2. docs/plans/0_plan_maestro.md (para identificar la fase)
3. Si existe código relacionado, leer los archivos
   principales para entender el estado actual

### Paso 2: Identificar la fase

Determina qué fase necesita especificación:
- Si el usuario dice "especifica la fase 2" → fase 2
- Si no indica fase → la primera fase sin spec.md

Verifica que la carpeta docs/plans/fase_X/ existe.
Si no existe, créala.

### Paso 3: Conversación de especificación

Haz las siguientes preguntas DE UNA EN UNA:

1. "¿Qué problema resuelve esta fase para el usuario final?"
   (la respuesta define la sección "Problema y objetivo")

2. "¿Quién interactúa con esta funcionalidad y qué hace?"
   (la respuesta define las historias de usuario)

3. "¿Cómo sabremos que está bien hecho? ¿Qué condiciones
    deben cumplirse?"
   (la respuesta define los criterios de aceptación)

4. "¿Qué datos maneja? ¿Qué entra y qué sale?"
   (la respuesta define los contratos de datos)

5. "¿Qué NO debe hacer esta fase? ¿Qué queda fuera?"
   (la respuesta define los anti-objetivos)

6. "¿Hay restricciones técnicas que yo deba saber?"
   (la respuesta define las restricciones — bases de datos
    existentes, APIs de terceros, limitaciones de rendimiento...)

Si la información ya está clara en el plan maestro o en
conversación previa, omite las preguntas que ya estén
respondidas. No hagas preguntas cuya respuesta ya conoces.

### Paso 4: Generar el borrador

Escribe el archivo X.spec.md con la estructura definida
más abajo. Preséntalo al usuario.

### Paso 5: Revisión y aprobación

Pregunta: "¿Hay algo incorrecto o que falte?"
Incorpora las correcciones.

Indica al usuario:
> "Especificación lista. Ejecuta /planear para crear
> el plan de implementación de esta fase."
```

---

## Plantilla del archivo de especificación (`X.spec.md`)

Esta plantilla está diseñada para ser rápida de rellenar (10–20 min) y aportar el máximo valor al agente.

```markdown
# Spec Fase X: [Nombre de la Fase]

> Especificación funcional — QUÉ construir y POR QUÉ.
> El CÓMO se define en el plan de fase (X.0_nombre.md).

---

## Problema y objetivo

**Problema**: [1-2 frases describiendo qué problema del usuario resuelve esta fase]

**Objetivo**: [1 frase: cómo luce el éxito al terminar esta fase]

---

## Historias de usuario

### HU-1: [Nombre descriptivo]
**Como** [tipo de usuario],
**quiero** [acción],
**para** [beneficio].

**Criterios de aceptación:**
- [ ] [condición medible y verificable]
- [ ] [condición medible y verificable]
- [ ] [condición medible y verificable]

### HU-2: [Nombre descriptivo]
...

---

## Contratos de datos

### [Modelo / Tabla / Esquema]
| Campo | Tipo | Restricciones |
|-------|------|---------------|
| ... | ... | ... |

### Endpoints (si aplica)

**[MÉTODO] [ruta]**
- Request: [esquema JSON o descripción breve]
- Response 200: [esquema JSON o descripción breve]
- Response error: [códigos y formato]

---

## Restricciones

- [Restricción técnica: BD existente, API externa, rendimiento...]
- [Restricción de negocio: regulación, formato de datos...]

---

## Anti-objetivos (lo que esta fase NO hace)

- NO [funcionalidad que podría confundirse con el alcance]
- NO [otra cosa que queda fuera]

---

## Contexto del código existente (si aplica)

- `ruta/archivo.ts` — [qué existe y qué relación tiene con esta spec]
- `ruta/otro.ts` — [dependencia relevante]
```

### Nivel de detalle según el tamaño de la fase

No todas las fases necesitan el mismo nivel de detalle. Ajusta según la complejidad:

| Tamaño de la fase | Tiempo en spec | Qué incluir | Qué omitir |
|---|---|---|---|
| **Pequeña** (1-2 días) | 5–10 min | Objetivo, 1-2 HU, criterios de aceptación, anti-objetivos | Contratos de datos completos (basta una descripción), contexto de código |
| **Media** (3-7 días) | 10–20 min | Todo lo de la plantilla | Se puede simplificar la sección de endpoints si la API es sencilla |
| **Grande** (1-2 semanas) | 20–30 min | Plantilla completa con detalle. Esquemas JSON explícitos. Todos los endpoints documentados | Nada — invertir aquí ahorra mucho después |

**Regla práctica**: si la fase toca más de 5 archivos, merece una spec con contratos de datos explícitos. Si toca 1-3 archivos, basta con historias de usuario y criterios de aceptación.

---

## Cómo cambia el flujo del plan de fase (`X.0`)

El plan de fase (`X.0_nombre.md`) no cambia de estructura, pero ahora se genera DESPUÉS de la spec y REFERENCIÁNDOLA. La plantilla del plan de fase se enriquece con una línea al principio:

```markdown
# Fase X: [Nombre de la Fase]

> Spec funcional: `docs/plans/fase_X/X.spec.md`
> Este plan implementa los requisitos definidos en la spec anterior.

## Objetivo
[Copiado o resumido de la spec — mantener alineación]

## Criterios de éxito
[Mapeados 1:1 desde los criterios de aceptación de la spec]
- [ ] [criterio de la spec]
- [ ] [criterio de la spec]

## Pasos de implementación
- [ ] Paso 1: [archivo + qué hacer]
- [ ] Paso 2: [archivo + qué hacer]

## Archivos afectados
- `ruta/archivo.ts` — qué cambia

## Notas / Decisiones técnicas
[Decisiones de diseño: por qué se eligió esta solución y no otra]

## Correctivos
[Referencias a fix-N si los hay]
```

La diferencia clave: los "Criterios de éxito" del plan ahora vienen directamente de los "Criterios de aceptación" de la spec. Hay una línea clara de trazabilidad: spec define QUÉ → plan define CÓMO → tareas ejecutan → tests verifican contra la spec.

---

## Integración con WAT

### ¿Cuándo se usa WAT y cuándo no?

WAT se activa cuando hay una tarea repetible o que requiere ejecución determinista (llamar a una API, transformar datos, desplegar). El proceso de especificación y planificación en sí NO necesita WAT — es trabajo de razonamiento puro del agente.

Pero hay un caso donde WAT y SDD se complementan directamente:

**Workflows que producen código deben leer la spec antes de ejecutar.**

Igual que tus workflows de marca leen `docs/brand/` antes de producir documentos, los workflows que generan código, configuraciones o infraestructura deben leer la spec de la fase activa. La regla para añadir a tu system prompt:

```markdown
## Regla de spec en workflows de código

Todo workflow que produzca o modifique código debe:
1. Leer CLAUDE.md (constitución del proyecto)
2. Leer la spec de la fase activa (docs/plans/fase_X/X.spec.md)
3. Verificar que el output cumple los criterios de aceptación de la spec

Si no existe spec para la fase activa, ejecutar /especificar primero.
```

### Nuevo workflow: `workflows/especificar.md`

Si quieres que `/especificar` funcione como un workflow WAT formal (para proyectos donde uses WAT), puedes crear este archivo:

```markdown
# Workflow: Especificar una fase

## Objetivo
Generar el archivo X.spec.md para una fase del plan maestro
mediante conversación con el usuario.

## Inputs requeridos
- Número de fase a especificar
- Acceso a CLAUDE.md y 0_plan_maestro.md

## Proceso
1. Leer CLAUDE.md y 0_plan_maestro.md
2. Identificar la fase objetivo
3. Ejecutar conversación de descubrimiento (preguntas 1-6 del protocolo)
4. Generar borrador de X.spec.md
5. Presentar al usuario para revisión
6. Incorporar correcciones y guardar archivo final

## Output
- Archivo docs/plans/fase_X/X.spec.md generado y aprobado

## Herramientas
- Ninguna (trabajo de razonamiento puro del agente)

## Siguiente paso
- Ejecutar /planear para la fase especificada
```

---

## Cambios en `/prime`: cargar specs en el contexto

El comando `/prime` necesita un ajuste menor para que el agente cargue las specs en su contexto al inicio de cada sesión. Añade este paso entre la Fase 3 y la Fase 4 actuales:

```markdown
### Fase 3b: Especificaciones funcionales

Para la fase activa del proyecto:
1. Leer X.spec.md si existe.
2. Anotar qué criterios de aceptación están pendientes vs completados.

Incluir en el informe de contexto:
- Si la fase activa tiene spec → "Spec: disponible (X criterios pendientes)"
- Si la fase activa NO tiene spec → "⚠️ Spec: no existe. Ejecutar /especificar."
```

---

## Cambios en el system prompt general (CLAUDE.md global)

Tu system prompt general de WAT necesita dos adiciones:

### 1. Añadir la regla de especificación al bloque de planificación

En la sección "Regla obligatoria: planificar antes de codificar", añade:

```markdown
## Regla obligatoria: especificar y planificar antes de codificar

**Antes de implementar cualquier cambio:**
1. Si es una fase nueva → ejecuta `/especificar` para definir QUÉ se construye
2. Luego ejecuta `/planear` para documentar CÓMO se construye
3. No toques código hasta que spec + plan estén escritos y confirmados

Excepciones (no requieren spec):
- Bug fixes puntuales (van directo a fix-N)
- Ajustes de una línea
- Ejecución de tools/workflows existentes sin modificar código

Excepciones (no requieren plan pero SÍ spec):
- Cambios pequeños que afectan a 1-2 archivos pero cambian
  comportamiento funcional
```

### 2. Añadir la estructura de specs a la documentación del proyecto

En la sección "Estructura de documentación", actualiza el árbol:

```
docs/plans/
├── 0_plan_maestro.md
├── fase_1/
│   ├── 1.spec.md              # Especificación funcional (QUÉ + POR QUÉ)
│   ├── 1.0_nombre_fase.md     # Plan de implementación (CÓMO)
│   └── 1.Y_desviacion.md      # Desviación o ajuste
├── fase_2/
│   └── ...
└── fixes/
    └── fix-N_nombre.md
```

Y añade la regla:

```markdown
**Spec funcional** (`X.spec.md`): Se crea ANTES del plan de fase.
Define QUÉ construir (historias de usuario, criterios de aceptación,
contratos de datos, anti-objetivos). El plan de fase lo implementa.
No se modifica durante la implementación salvo que cambie el alcance
(en cuyo caso, primero se actualiza la spec, luego el plan).
```

---

## Resumen visual del proceso completo

```
PROYECTO NUEVO
══════════════

  ┌──────────────────┐
  │ /crea-plan-maestro│  ← Conversación: qué problema, para quién, fases
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Scaffold proyecto │  ← npm create, pip init, etc.
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ /init-project     │  ← Lee código real → genera CLAUDE.md
  │ (mejorado)        │     + permisos del agente + anti-patrones
  └────────┬─────────┘
           │
           ▼

PARA CADA FASE (repetir)
════════════════════════

  ┌──────────────────┐
  │ /especificar      │  ← Conversación: qué, por qué, criterios,
  │ (NUEVO)           │     datos, restricciones, anti-objetivos
  │                   │  → Genera X.spec.md
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ /planear          │  ← Lee spec → genera plan de implementación
  │ (mejorado)        │  → Genera X.0_nombre.md con pasos + archivos
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Implementar       │  ← El agente ejecuta tareas del plan
  │                   │     (usa WAT si hay workflows/tools relevantes)
  │                   │     Si aparece desviación → X.Y + actualizar X.0
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Verificar         │  ← Tests + comprobar criterios de aceptación
  │                   │     de X.spec.md uno por uno
  │                   │  → Marcar [x] en plan maestro
  └────────┬─────────┘
           │
           ▼
        ¿Más fases? ─── Sí ──→ volver a /especificar
           │
           No
           │
           ▼
        Proyecto completado


CADA NUEVA SESIÓN
═════════════════

  ┌──────────────────┐
  │ /prime            │  ← Carga CLAUDE.md + plan maestro + spec activa
  │ (mejorado)        │     + plan de fase + git log
  └──────────────────┘
```

---

## Ejemplo práctico: proyecto de 50 horas

Para aterrizar todo esto, veamos cómo sería el flujo en un proyecto real del tamaño que describes.

### El proyecto

Un sistema de gestión de citas online para una clínica dental. 50 horas estimadas.

### Paso 0: `/crea-plan-maestro` (10 min)

Resultado: `0_plan_maestro.md` con 4 fases:
- Fase 1: Gestión de citas (crear, ver, cancelar)
- Fase 2: Gestión de pacientes (perfil, historial)
- Fase 3: Notificaciones (recordatorios por email)
- Fase 4: Panel de métricas (ocupación, no-shows)

### Paso 1: Scaffold (5 min)

```bash
npx create-next-app@latest clinica-dental --typescript --tailwind
```

### Paso 2: `/init-project` (10 min)

Resultado: `CLAUDE.md` con el stack real, comandos, y estas secciones nuevas:

```
✅ SIEMPRE: ejecutar npm test, usar TypeScript estricto, Prisma para BD
⚠️ PREGUNTAR: cambios en schema.prisma, nuevas dependencias
🚫 NUNCA: SQL directo sin Prisma, lógica en API routes, secretos en código
```

### Paso 3: `/especificar` para Fase 1 (15 min)

Conversación rápida de 6 preguntas → resultado `1.spec.md`:

```markdown
# Spec Fase 1: Gestión de Citas

## Problema y objetivo

**Problema**: Los pacientes de la clínica llaman por teléfono para pedir cita,
generando colas y errores de agenda. La recepcionista gestiona todo en papel.

**Objetivo**: Los pacientes pueden reservar, ver y cancelar citas online.
La agenda se actualiza en tiempo real.

## Historias de usuario

### HU-1: Reservar cita
**Como** paciente registrado,
**quiero** elegir fecha, hora y dentista para mi cita,
**para** reservarla sin llamar por teléfono.

**Criterios de aceptación:**
- [ ] Solo se muestran huecos disponibles (no los ocupados)
- [ ] Al confirmar, se crea la cita en BD con estado "confirmada"
- [ ] No se pueden reservar citas en el pasado
- [ ] No se pueden reservar dos citas en el mismo hueco

### HU-2: Ver mis citas
**Como** paciente registrado,
**quiero** ver una lista de mis citas futuras,
**para** saber cuándo tengo que ir.

**Criterios de aceptación:**
- [ ] Se muestran solo citas futuras, ordenadas por fecha
- [ ] Cada cita muestra: fecha, hora, dentista, estado

### HU-3: Cancelar cita
**Como** paciente registrado,
**quiero** cancelar una cita con al menos 24h de antelación,
**para** liberar el hueco para otro paciente.

**Criterios de aceptación:**
- [ ] Solo se puede cancelar con ≥24h de antelación
- [ ] Al cancelar, el hueco vuelve a estar disponible
- [ ] La cita pasa a estado "cancelada" (no se borra)

## Contratos de datos

### Tabla: appointments
| Campo | Tipo | Restricciones |
|-------|------|---------------|
| id | UUID | PK |
| patient_id | UUID | FK → users.id, NOT NULL |
| dentist_id | UUID | FK → dentists.id, NOT NULL |
| date | DATE | NOT NULL, >= today |
| time_slot | TIME | NOT NULL |
| status | ENUM('confirmed','cancelled','completed') | DEFAULT 'confirmed' |
| created_at | TIMESTAMP | DEFAULT now() |

**Invariante**: No pueden existir dos citas con mismo dentist_id + date + time_slot
donde status = 'confirmed'.

### Endpoints

**GET /api/appointments/available?dentist_id=X&date=Y**
- Response 200: `{ slots: ["09:00", "09:30", "10:00", ...] }`

**POST /api/appointments**
- Request: `{ dentist_id, date, time_slot }`
- Response 201: `{ appointment: { id, date, time_slot, status } }`
- Response 409: `{ error: "Slot already booked", code: "SLOT_TAKEN" }`

**DELETE /api/appointments/:id**
- Response 200: `{ appointment: { id, status: "cancelled" } }`
- Response 422: `{ error: "Cannot cancel within 24h", code: "TOO_LATE" }`

## Restricciones

- Base de datos: PostgreSQL con Prisma ORM
- Los time_slots son fijos: cada 30 min de 09:00 a 18:00
- Autenticación: NextAuth con provider de email

## Anti-objetivos

- NO incluye pagos online (eso sería fase futura)
- NO envía confirmación por email (eso es fase 3)
- NO gestiona tipos de tratamiento (solo "cita genérica")
```

**Tiempo total: 15 minutos.** Y ahora el agente sabe exactamente qué construir, con qué estructura de datos, qué endpoints, y qué está fuera de alcance.

### Paso 4: `/planear` (5 min)

El agente lee `1.spec.md` y genera `1.0_gestion_citas.md` con pasos de implementación concretos, archivos afectados, y criterios de éxito mapeados 1:1 desde la spec.

### Pasos 5-6: implementar y verificar

El agente ejecuta las tareas del plan. Al terminar, se comprueban los criterios de aceptación de la spec uno por uno. Los que pasan se marcan [x]. Los que no, se documentan como desviaciones.

**Resultado neto**: 30 minutos de documentación (plan maestro + CLAUDE.md + spec + plan de fase) para una fase de 15-20 horas de implementación. Esos 30 minutos ahorran fácilmente 3-5 horas de iteraciones con el agente por malentendidos, funcionalidades inventadas y bugs de especificación.

---

## Cuándo NO usar spec completa

Para mantener la agilidad en proyectos pequeños:

| Situación | Qué hacer |
|-----------|-----------|
| **Bug fix puntual** | Solo `fix-N`. Sin spec, sin plan de fase. |
| **Cambio de 1 línea** | Nada. Hazlo directamente. |
| **Fase trivial** (< 4h de trabajo, 1-2 archivos) | Spec mínima: solo objetivo + criterios de aceptación. Sin contratos de datos ni endpoints. |
| **Fase media** (4-20h, 3-10 archivos) | Spec estándar: la plantilla completa. |
| **Fase grande** (> 20h, > 10 archivos) | Spec completa con esquemas JSON, todos los endpoints documentados, diagramas de flujo. |
| **Prototipo / exploración** | Sin spec. Haz vibe coding. Cuando funcione, crea la spec retroactivamente antes de pasar a producción. |

---

## Checklist de integración

Para integrar SDD en tu framework actual, los cambios concretos son:

### Archivos nuevos a crear
- [ ] `workflows/especificar.md` — el protocolo del comando /especificar

### Archivos a modificar
- [ ] System prompt general (CLAUDE.md global) — añadir regla de "especificar antes de planificar" y estructura de specs
- [ ] `workflows/init-project.md` — añadir paso 4b (permisos + anti-patrones)
- [ ] `workflows/planear.md` — añadir paso 2b (leer spec antes de crear plan)
- [ ] `workflows/prime.md` — añadir fase 3b (cargar spec activa)

### Nada que cambiar
- [ ] `/crea-plan-maestro` — sin cambios
- [ ] Estructura de desviaciones (X.Y) — sin cambios
- [ ] Estructura de fixes (fix-N) — sin cambios
- [ ] WAT core (herramienta separada) — sin cambios
- [ ] docs/brand/, docs/learnings/, docs/refs/ — sin cambios
