# Referencia rápida de comandos

## `/prime`
**Cuándo usarlo:** Al inicio de cada sesión, antes de cualquier tarea.

Carga el contexto completo del proyecto: lee `CLAUDE.md`, el plan maestro, la spec y plan de fase activos, el historial de git y los archivos principales del código. Entrega un informe con la fase activa, estado de la spec, tareas pendientes, últimos commits y el próximo paso sugerido.

---

## `/1-crea-plan-maestro`
**Cuándo usarlo:** Al arrancar un proyecto nuevo, antes de escribir código.

Conduce una conversación para entender qué se construye, para quién y en qué fases. Usa la plantilla `docs/templates/0_plan_maestro.md`.

**Genera:** `docs/plans/0_plan_maestro.md`

---

## `/2-scaffold`
**Cuándo usarlo:** Tras crear el plan maestro, antes de `/3-init-project`.

Crea la estructura base del proyecto ejecutando los comandos de scaffold del ecosistema elegido (npm create, uv init, etc.). Pregunta por tipo de proyecto y stack. Crea las carpetas del framework (`docs/`, `tests/`, etc.).

**Genera:** Proyecto inicializado con código real

---

## `/3-init-project`
**Cuándo usarlo:** Una vez que el proyecto tiene código (después del scaffold).

Lee el código real del proyecto (`package.json`, archivos principales, `.env.example`, configuraciones) y genera el `CLAUDE.md` con la arquitectura, stack, comandos, convenciones, permisos del agente, anti-patrones y gotchas reales. Usa la plantilla `docs/templates/CLAUDE_proyecto.md`.

**Requiere que exista código** — no se puede generar desde cero.

**Genera:** `CLAUDE.md` en la raíz del proyecto

---

## `/4-especificar`
**Cuándo usarlo:** Antes de empezar cada fase nueva.

Genera la especificación funcional (`X.spec.md`) para una fase del plan maestro. Define QUÉ construir y POR QUÉ mediante una conversación de descubrimiento con el usuario. Usa la plantilla `docs/templates/X.spec.md`.

**Genera:** `docs/plans/fase_X/X.spec.md`

---

## `/5-planear`
**Cuándo usarlo:** Después de la spec, antes de codificar.

Lee la spec de la fase, genera el plan de implementación y lo presenta para aprobación. Usa las plantillas de `docs/templates/`.

**Genera:**
- `docs/plans/fase_X/X.0_nombre.md` — plan principal de la fase
- `docs/plans/fase_X/X.tasks.md` — tareas atómicas (opcional, si >10 pasos)

---

## `/6-implementar`
**Cuándo usarlo:** Después de que el plan esté aprobado, para ejecutar los pasos.

Ejecuta los pasos del plan de fase siguiendo el ciclo test-first: escribir test, verificar que falla, escribir código, verificar que pasa, marcar paso `[x]`. Gestiona desviaciones si aparecen.

**Modos:**
- `/6-implementar` — ejecuta todos los pasos pendientes
- `/6-implementar 3` — ejecuta solo el paso 3
- `/6-implementar 3-5` — ejecuta los pasos 3 a 5
- `/6-implementar siguiente` — ejecuta el siguiente paso pendiente

---

## `/7-verificar`
**Cuándo usarlo:** Al terminar una fase, o a mitad de fase para comprobar progreso.

Verifica la alineación entre la spec y el código: ejecuta los tests del proyecto, comprueba que cada criterio de aceptación tiene test y pasa, compara los contratos de datos con el código real y revisa que no se han implementado anti-objetivos. Genera un informe con el estado de cada criterio.

**Genera:** Informe de verificación (no crea archivo, muestra resultado en consola)

---

## `/8-auditar`
**Cuándo usarlo:** Al cerrar cada fase (tras `/7-verificar` y antes de `/9-documentar`), antes de cualquier release, tras un fix crítico o cuando se pida una revisión de seguridad.

Revisión de seguridad autocontenida (no usa skills). Acota el alcance a la fase, ejecuta los escáneres instalados, lee el código buscando las clases de fallo que los escáneres no ven, presenta los hallazgos ordenados por gravedad con `file:line` y CWE, y los arregla en la misma sesión con su test de regresión.

Incluye **escape hatch**: si la fase activa no toca código de seguridad relevante (solo docs, assets o refactor puro sin cambio de comportamiento), el comando pregunta si saltar la auditoría y registra la exención en el catálogo.

**Modos:**
- `/8-auditar` — audita los archivos de la **fase activa** (por defecto)
- `/8-auditar fase X` — audita una fase concreta
- `/8-auditar completo` — audita todo `src/` (release gate)
- `/8-auditar deps` — solo dependencias
- `/8-auditar secretos` — solo búsqueda de credenciales hardcoded
- `/8-auditar <ruta>` — scope restringido a ruta/archivo/glob

**Genera:** hallazgos en chat + código corregido + una línea por hallazgo en `docs/security/README.md`

---

## `/9-documentar`
**Cuándo usarlo:** Al terminar una fase (después de `/8-auditar`) o al final del proyecto.

Genera la guía de usuario de la aplicación (`docs/GUIA_USUARIO.md`). Lee las specs completadas, convierte las historias de usuario en instrucciones prácticas y verifica contra el código real. Documentación breve, con bullet points y emojis, para usuarios no técnicos.

**Genera:** `docs/GUIA_USUARIO.md`

---

## Flujo completo

```
PROYECTO NUEVO
1. /1-crea-plan-maestro  → define qué construyes y en qué fases
2. /2-scaffold           → crea el proyecto (npm create, uv init, etc.)
3. /3-init-project       → genera CLAUDE.md desde el código real

CADA FASE
4. /4-especificar        → define QUÉ construir (spec funcional)
5. /5-planear            → define CÓMO implementar (plan de fase)
6. /6-implementar        → código + tests (ciclo test-first por paso)
7. /7-verificar          → alineación spec ↔ código + marcar [x]
8. /8-auditar            → auditoría de seguridad (fase o completo)
9. /9-documentar         → guía de usuario (al terminar fase o proyecto)

RELEASE GATE (antes de empaquetar / desplegar / publicar)
/8-auditar completo      → auditoría full-project para cazar issues cruzados

CADA SESIÓN
/prime                   → carga contexto antes de trabajar
```
