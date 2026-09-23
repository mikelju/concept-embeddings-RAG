# Comando /4-especificar

Genera el archivo de especificación funcional (`X.spec.md`) para una fase del plan maestro.
Se ejecuta ANTES de `/5-planear` para definir QUÉ construir y POR QUÉ, sin entrar en el CÓMO.

---

## Cuándo usar este comando

- **Fase nueva**: antes de empezar a implementar cualquier fase del plan maestro.
- **Cambio de alcance**: cuando los requisitos de una fase han cambiado y la spec anterior ya no refleja la realidad.
- **Spec retroactiva**: cuando se ha implementado algo sin spec (vibe coding, prototipo) y se quiere formalizar antes de seguir.

---

## Protocolo

### Paso 1: Cargar contexto

Lee en este orden:

1. **`CLAUDE.md`** → ya está en contexto (se carga solo); no lo releas. Las reglas de protocolo experimental están en la skill `research-protocol`.
2. **`docs/plans/0_master_plan.md`** → identificar la fase y su descripción actual.
3. **Código existente relacionado** (si lo hay) → leer archivos principales que afecten a esta fase para no especificar cosas que ya existen o contradecir la realidad del código.

### Paso 2: Identificar la fase

- Si el usuario dice "especifica la fase 3" → fase 3.
- Si no indica fase → la primera fase que NO tenga archivo `X.spec.md`.
- Verificar que la carpeta `docs/plans/phase_X/` existe. Si no existe, crearla.

### Paso 3: Conversación de especificación

Haz las siguientes preguntas **de una en una**, esperando respuesta antes de continuar.
No hagas todas a la vez — el objetivo es una conversación, no un formulario.

1. **"¿Qué problema resuelve esta fase para el usuario final?"**
   → Define la sección "Problema y objetivo"

2. **"¿Quién interactúa con esta funcionalidad y qué puede hacer?"**
   → Define las historias de usuario

3. **"¿Cómo sabremos que está bien hecho? ¿Qué condiciones deben cumplirse?"**
   → Define los criterios de aceptación

4. **"¿Qué datos maneja? ¿Qué entra y qué sale?"**
   → Define los contratos de datos (modelos, endpoints, formatos)

5. **"¿Qué NO debe hacer esta fase? ¿Qué queda fuera?"**
   → Define los anti-objetivos

6. **"¿Hay restricciones técnicas que deba saber?"**
   → Define restricciones (BD existente, API externa, rendimiento, regulación...)

**Reglas de la conversación:**
- Si la información ya está clara en el plan maestro, en CLAUDE.md o en la conversación previa, NO repitas la pregunta. Usa lo que ya sabes.
- Si el usuario da respuestas cortas, propón detalles tú y pide confirmación en lugar de forzar respuestas largas.
- Si la fase es pequeña (< 4h de trabajo), simplifica: solo preguntas 1, 3 y 5. Las demás se infieren.

### Paso 4: Generar el borrador

Lee la plantilla `docs/templates/X.spec.md` y úsala como base para generar el archivo `docs/plans/phase_X/X.spec.md`.

**Nivel de detalle según tamaño de la fase:**

| Tamaño de fase | Qué incluir |
|---|---|
| **Pequeña** (< 1 día) | Objetivo + criterios de aceptación + anti-objetivos. Sin contratos detallados. |
| **Media** (1-5 días) | Plantilla completa. Contratos de datos con campos principales. |
| **Grande** (> 5 días) | Plantilla completa con detalle máximo. Esquemas JSON explícitos. Todos los endpoints. |

**Regla práctica**: si la fase toca >5 archivos, merece contratos de datos explícitos. Si toca 1-3 archivos, basta con historias de usuario y criterios de aceptación.

### Paso 5: Revisión y aprobación

Presenta el borrador completo al usuario y pregunta:
> "¿Hay algo incorrecto, que falte o que sobre?"

Incorpora las correcciones. El archivo queda bloqueado: no se modifica durante la implementación salvo que cambie el alcance de la fase.

### Paso 6: Siguiente paso

Tras guardar el archivo, indica:
> "Spec de la Fase X lista. Ejecuta `/5-planear` para crear el plan de implementación."
