# Comando /1-crea-plan-maestro

Genera el archivo `docs/plans/0_plan_maestro.md` para un proyecto nuevo o lo actualiza si ya existe.
Este es el **primer comando a ejecutar** en cualquier proyecto nuevo, antes de escribir una sola línea de código.

---

## Regla fundamental

**El plan maestro solo se actualiza después de probar y verificar que las cosas funcionan.** No se toca al crear tools, planificar o escribir código — solo cuando se ha ejecutado el trabajo y se ha confirmado que funciona correctamente. El plan maestro refleja estado real verificado, no intenciones.

## Cuándo usar este comando

- **Proyecto nuevo**: tienes una idea pero aún no hay código. Este comando define qué vas a construir.
- **Proyecto existente sin plan**: ya hay código pero falta documentar el roadmap.
- **Revisión del roadmap**: el producto ha evolucionado y el plan maestro está desactualizado. Solo actualizar con cambios ya probados.

---

## Protocolo

### Paso 1: Recopilar información existente

Antes de hacer preguntas, comprueba si ya existe algún contexto:

- ¿Existe `docs/plans/0_plan_maestro.md`? → léelo para no repetir lo que ya está definido.
- ¿Existe `CLAUDE.md`? → léelo para conocer el stack y la arquitectura ya decididos.
- ¿Existe `package.json`? → léelo para inferir el stack tecnológico real.
- ¿Hay commits en git? → ejecuta `git log --oneline -5` para entender qué se ha construido ya.

### Paso 2: Conversación de descubrimiento

Haz las siguientes preguntas **de una en una**, esperando respuesta antes de continuar.
No hagas todas a la vez — el objetivo es una conversación, no un formulario.

1. **¿Qué problema resuelve este producto?** (en una frase)
2. **¿Quién lo usa?** (tipo de usuario, empresa o persona)
3. **¿Cuál es el MVP mínimo que ya funcionaría?** (qué tiene que hacer para ser útil el primer día)
4. **¿Cuál es la visión a largo plazo?** (dónde quieres llegar en 6-12 meses)
5. **¿Qué stack tecnológico usarás?** (si no lo sabe aún, propón opciones basadas en el tipo de proyecto)
6. **¿Hay fases naturales en la evolución del producto?** (ej. primero funcionalidad core, luego multi-usuario, luego pagos)

Si el proyecto ya tiene código o `CLAUDE.md`, omite las preguntas cuya respuesta ya conoces.

### Paso 3: Proponer las fases

Basándote en las respuestas, propón un desglose en fases con este criterio:
- Cada fase debe poder usarse de forma independiente (no depender de la siguiente para tener valor).
- La Fase 1 siempre debe ser el MVP funcional más pequeño posible.
- Máximo 6-7 fases. Si hay más, agrupa.

Presenta la propuesta de fases al usuario y **pide confirmación antes de escribir el archivo**.

### Paso 4: Generar el archivo

Lee la plantilla `docs/templates/0_plan_maestro.md` y úsala como base para generar `docs/plans/0_plan_maestro.md`. Rellena las fases confirmadas por el usuario.

### Paso 5: Siguiente paso

Tras crear el archivo, indica al usuario:

> "El plan maestro está listo. Los siguientes pasos son:
> 1. **Crear el scaffold del proyecto** (ej. `npm create vite@latest`) si aún no hay código
> 2. **`/3-init-project`** para generar el `CLAUDE.md` a partir del código real
> 3. **`/4-especificar`** para crear la spec funcional de la Fase 1
> 4. **`/5-planear`** para crear el plan de implementación de la Fase 1"
