# Skill: Protocolo de Planificación y Documentación

Este skill define cómo actuar cuando se propone cualquier cambio, mejora o corrección en el proyecto.

## Regla de oro

**Antes de tocar código, hay un plan documentado.** Si no hay plan, se crea. Si hay desviación, se registra. Si la fase no tiene spec, se ejecuta `/4-especificar` primero.

---

## Estructura de documentación

```
docs/plans/
├── 0_plan_maestro.md          # Visión global de todas las fases
├── fase_1/
│   ├── 1.spec.md              # Especificación funcional (QUÉ + POR QUÉ)
│   ├── 1.0_nombre_fase.md     # Plan de implementación (CÓMO)
│   ├── 1.tasks.md             # (Opcional) Tareas atómicas para fases complejas
│   ├── 1.1_problema_X.md      # Desviación o problema encontrado (correlativo)
│   └── 1.2_ajuste_Y.md        # Otro ajuste fuera del plan principal
├── fase_2/
│   ├── 2.spec.md
│   ├── 2.0_nombre_fase.md
│   └── ...
└── fixes/
    ├── fix-1_nombre.md        # Bug fix o correctivo puntual (correlativo global)
    └── fix-2_nombre.md
```

### Nomenclatura de archivos

| Archivo | Cuándo usarlo |
|---|---|
| `X.spec.md` | Especificación funcional. Se crea con `/4-especificar` ANTES del plan de fase. |
| `X.0_nombre.md` | Plan principal de la fase X. Se crea al iniciar la fase, DESPUÉS de la spec. |
| `X.tasks.md` | Tareas atómicas. OPCIONAL — solo si la fase tiene >10 pasos. |
| `X.Y_nombre.md` | Desviación, problema o ajuste fuera del plan principal. Y es correlativo (1, 2, 3...). |
| `fixes/fix-N_nombre.md` | Bug fix o correctivo puntual entre fases o dentro de una fase. N es correlativo global. |

---

## Cuándo actualizar cada documento

### Spec Funcional (`X.spec.md`)
- Se crea ANTES del plan de fase, mediante `/4-especificar`.
- **NO se modifica** durante la implementación salvo que cambie el alcance de la fase.
- Si el alcance cambia → actualizar primero la spec, luego propagar al plan de fase.

### Plan Maestro (`0_plan_maestro.md`)
- Cuando se **añade una fase nueva** que no existía.
- Cuando se **completa un hito** de una fase (marcar `[x]`).
- Cuando el **alcance de una fase cambia** sustancialmente.
- NO para detalles de implementación — eso va en el plan de fase.

### Plan de Fase (`X.0_nombre.md`) — fuente de verdad de la fase
- Hay **exactamente uno por fase** (`X.0_*.md`). Es el equivalente al plan maestro pero en el contexto de esa fase concreta.
- Se crea **después de la spec** y referenciándola. Los criterios de éxito se mapean 1:1 desde los criterios de aceptación de la spec.
- Contiene: objetivo, pasos concretos, criterios de éxito, archivos afectados, notas técnicas.
- **Se mantiene siempre actualizado** como reflejo fiel del estado real de la fase:
  - Marcar tareas completadas (`[x]`) conforme avanza el trabajo.
  - Si una desviación (`X.Y`) cambia el enfoque, los pasos o el alcance → actualizar el `X.0` para que refleje la decisión adoptada.
  - Si se descubren notas técnicas, gotchas o decisiones de diseño → añadirlas al `X.0`.
- **Regla de propagación**: cualquier cambio sustancial en el plan de fase (no simples `[x]`) debe reflejarse también en la descripción de esa fase dentro del plan maestro.

### Tareas (`X.tasks.md`) — OPCIONAL
- Solo se crea cuando la fase tiene **más de 10 pasos de implementación**.
- Descompone el plan en tareas atómicas: cada tarea = 1 commit, máximo 3-5 archivos.
- Si no existe, los pasos del plan de fase (`X.0`) cumplen esa función.

### Documento de Desviación (`X.Y_nombre.md`)
- Se crea cuando:
  - Aparece un **problema inesperado** que bloquea o altera el plan.
  - Hay que hacer un **cambio fuera del plan** de la fase actual.
  - Un fix o solución requiere **más de 2-3 pasos no triviales**.
- Contiene: descripción del problema, causa raíz, solución adoptada, impacto.
- **Obligación de propagación**: tras crear o cerrar una desviación, actualizar el plan de fase (`X.0`) para que refleje el cambio. La desviación es el registro histórico del "qué pasó"; el plan de fase es el reflejo del "cómo queda".

### Fix o correctivo puntual (`fixes/fix-N_nombre.md`)
- Se crea cuando aparece un **bug o comportamiento incorrecto** en funcionalidad ya existente,
  independientemente de en qué fase se encuentre el proyecto.
- N es un número **correlativo global** (fix-1, fix-2...), no depende de la fase.
- **Regla de correlación obligatoria**: el fix siempre debe quedar referenciado en el plan de la
  fase a la que pertenece la funcionalidad afectada:
  - Si la fase tiene plan doc (`X.0_*.md`) → añadir entrada en la sección "Correctivos" de ese doc.
  - Si la fase no tiene plan doc → añadir entrada en la sección "Correctivos" del plan maestro.
- Contiene: descripción del bug, causa raíz, solución adoptada, archivos modificados.

---

## Protocolo paso a paso al recibir una nueva tarea

1. **Leer el plan maestro** (`docs/plans/0_plan_maestro.md`) para entender el contexto.
2. **Determinar a qué fase pertenece** la tarea solicitada.
   - Si encaja en una fase existente → continuar.
   - Si es algo nuevo → proponer añadirla como nueva fase al plan maestro.
3. **Verificar si existe la spec funcional** (`docs/plans/fase_X/X.spec.md`).
   - Si NO existe y la tarea lo requiere → comunicar: "No hay especificación funcional para esta fase. Ejecuta `/4-especificar` primero para definir QUÉ se construye." No continuar hasta que exista (o el usuario decida saltársela explícitamente).
   - Si existe → leerla para conocer los criterios de aceptación y contratos.
4. **Verificar si existe el plan de fase** (`docs/plans/fase_X/X.0_nombre.md`).
   - Si no existe → crearlo ahora, referenciando la spec. Los criterios de éxito deben mapearse desde los criterios de aceptación de la spec.
   - Si existe → leerlo para entender el alcance y el estado.
5. **Presentar el plan al usuario** para revisión y aprobación. No se implementa nada hasta que el usuario confirme.
6. **Indicar el siguiente paso**:
   > "Plan de fase listo. Ejecuta `/6-implementar` para empezar la implementación, o `/6-implementar siguiente` para ir paso a paso."

---

## Plantillas

Todos los documentos que genera este protocolo usan las plantillas de `docs/templates/`:

| Documento | Plantilla |
|-----------|-----------|
| Plan de fase | `docs/templates/X.0_plan_fase.md` |
| Tareas (opcional, si >10 pasos) | `docs/templates/X.tasks.md` |
| Desviación | `docs/templates/X.Y_desviacion.md` |
| Fix | `docs/templates/fix-N_nombre.md` |

Lee la plantilla correspondiente antes de generar cada documento.

---

## Ahora: aplica el protocolo

Lee el plan maestro y los planes de fase existentes, determina en qué punto está el proyecto y comunica al usuario:
1. El estado actual de cada fase (pendiente / en curso / completada).
2. Si la fase activa tiene spec (`X.spec.md`) o no.
3. Qué sería el siguiente paso lógico según el plan.
4. Si la tarea solicitada encaja en el plan o requiere actualización del mismo.
