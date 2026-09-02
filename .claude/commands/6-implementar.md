# Comando /6-implementar

Ejecuta los pasos del plan de fase siguiendo el ciclo test-first: escribir test, verificar que falla, escribir código, verificar que pasa, marcar paso completado.

---

## Modos de uso

```
/6-implementar              → ejecuta todos los pasos pendientes de la fase activa
/6-implementar 3            → ejecuta solo el paso 3
/6-implementar 3-5          → ejecuta los pasos 3, 4 y 5
/6-implementar siguiente    → ejecuta el siguiente paso pendiente
```

---

## Protocolo

### Paso 1: Cargar contexto

Lee en este orden:

1. **`CLAUDE.md`** → reglas del proyecto, comando de tests, convenciones, permisos.
2. **`docs/plans/fase_X/X.spec.md`** → criterios de aceptación y contratos de datos.
3. **`docs/plans/fase_X/X.0_nombre.md`** → pasos de implementación y estado actual.
4. **`docs/plans/fase_X/X.tasks.md`** (si existe) → tareas atómicas.

Si no existe spec → comunicar:
> "No hay spec para esta fase. Ejecuta `/4-especificar` primero."

Si no existe plan de fase → comunicar:
> "No hay plan para esta fase. Ejecuta `/5-planear` primero."

### Paso 2: Determinar qué pasos ejecutar

Según el modo de invocación:

- **Sin argumento**: todos los pasos marcados como `[ ]` en el plan de fase.
- **Número** (`3`): solo ese paso. Verificar que existe y que está pendiente.
- **Rango** (`3-5`): los pasos del 3 al 5. Verificar que existen.
- **`siguiente`**: el primer paso marcado como `[ ]`.

Comunicar al usuario qué pasos se van a ejecutar antes de empezar:
> "Voy a ejecutar los pasos X, Y, Z del plan de fase. ¿Confirmas?"

### Paso 3: Ejecutar cada paso (ciclo test-first)

Para cada paso a ejecutar, seguir este ciclo:

**3a. Identificar el criterio de aceptación.**
¿Qué criterio de la spec cubre este paso? Si el paso no está vinculado a ningún criterio, anotarlo.

**3b. Escribir el test.**
Crear el test que verifica el criterio de aceptación. El test debe:
- Estar en la carpeta de tests del proyecto (según estructura de `CLAUDE.md`).
- Nombrar claramente qué criterio verifica.
- Ser ejecutable con el comando de tests del proyecto.

**3c. Verificar que el test falla (red).**
Ejecutar el comando de tests. El test nuevo DEBE fallar porque el código aún no existe.
- Si el test pasa sin código → el test no verifica nada útil. Reescribirlo.
- Si otros tests que antes pasaban ahora fallan → detenerse, hay un problema. Investigar antes de continuar.

**3d. Escribir el código.**
Implementar el código que hace pasar el test, siguiendo:
- Las convenciones de `CLAUDE.md`.
- Los contratos de datos de la spec (campos, tipos, restricciones).
- Los anti-patrones prohibidos.

**3e. Verificar que el test pasa (green).**
Ejecutar el comando de tests.
- Si el test nuevo pasa y los demás siguen pasando → continuar.
- Si el test nuevo falla → corregir el código y volver a ejecutar.
- Si otros tests que antes pasaban ahora fallan → detenerse. Algo se ha roto. Corregir antes de continuar.

**3f. Marcar el paso como completado.**
- Marcar `[x]` en el plan de fase (`X.0_nombre.md`).
- Si existe `X.tasks.md`, marcar también allí.

**3g. Comunicar progreso.**
> "Paso X completado. Test pasa. [Breve descripción de lo implementado]."

### Paso 4: Gestionar desviaciones

Si durante la ejecución de un paso aparece un problema inesperado:

1. **Detenerse** — no seguir con el siguiente paso.
2. **Comunicar** al usuario qué ha pasado y qué impacto tiene.
3. **Crear desviación** (`X.Y_nombre.md`) usando la plantilla `docs/templates/X.Y_desviacion.md`.
4. **Esperar confirmación** del usuario si el impacto es significativo.
5. **Actualizar el plan de fase** para reflejar la decisión adoptada.
6. Si el cambio altera los requisitos → **actualizar la spec** y propagar.
7. Retomar la ejecución cuando el problema esté resuelto.

### Paso 5: Al terminar

Cuando se hayan ejecutado todos los pasos solicitados:

1. **Actualizar el plan de fase**: verificar que todos los pasos ejecutados están marcados `[x]`.
2. **Comunicar resumen**:

```
## Resumen de implementación

### Pasos ejecutados
- [x] Paso 3: [descripción] — test: `tests/ruta/test_archivo.py`
- [x] Paso 4: [descripción] — test: `tests/ruta/test_archivo.py`
- [x] Paso 5: [descripción] — test: `tests/ruta/test_archivo.py`

### Estado de la fase
[X de Y pasos completados]

### Tests
[N tests nuevos creados, todos pasan]

### Siguiente acción
[Siguiente paso pendiente | Ejecutar /7-verificar si la fase está completa]
```

3. Si **todos los pasos de la fase están completados** → sugerir:
   > "Todos los pasos del plan están completados. Ejecuta `/7-verificar` para comprobar la alineación spec ↔ código antes de cerrar la fase."
