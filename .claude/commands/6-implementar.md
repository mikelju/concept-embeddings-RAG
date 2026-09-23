# Comando /6-implementar

Ejecuta el plan aprobado de principio a fin **sin pedir confirmación paso a paso**. El autor revisa
al final, en la PR. Su atención está en los dos extremos: la spec antes y la PR después.

---

## Modos de uso

```
/6-implementar              → todos los pasos pendientes del plan activo, y entrega
/6-implementar 3            → solo el paso 3 (sin entregar)
/6-implementar 3-5          → pasos 3 a 5 (sin entregar)
/6-implementar siguiente    → el siguiente paso pendiente (sin entregar)
```

---

## Precondiciones

- Existe una spec **aprobada** y un plan de fase en `docs/plans/phase_X/`. Si la spec está en
  borrador o falta el plan, detente y dilo: no se implementa sobre una spec no aprobada.
- Estás en una rama de trabajo, nunca en `main`. Si estás en `main`, crea la rama
  (`phase-X-<nombre>` o `<fase>-<cambio>`) antes de tocar nada.

## Paso 1: Cargar contexto

1. **`CLAUDE.md`** → ya está en contexto (se carga solo); no lo releas. Las reglas de protocolo experimental están en la skill `research-protocol`.
2. Carga la skill `research-protocol`.
3. Lee la spec y el plan `X.0_*.md` de la fase. Del plan, lee con detalle solo los pasos que vas
   a ejecutar.

## Paso 2: Ejecutar cada paso

Para cada paso pendiente, en orden:

1. **Test donde importa.** Escribe primero un test para el código que puede cambiar un resultado
   experimental (métricas, rankings, splits, fusión, cachés, selección). Compruébalo en rojo y luego
   en verde. No hace falta un test por cada invariante de proceso: es la regla de simplificación.
2. **Implementa** siguiendo el plan y las convenciones de `CLAUDE.md`.
3. **Comprueba**: los tests del área tocada, y `uv run ruff check .` en los ficheros cambiados.
4. **Marca** `[x]` en el plan.
5. **Commit** del paso con rutas explícitas (`git add <rutas>`, nunca `git add -A`), mensaje en
   inglés que nombre el paso. Haz `git push` de la rama cuando convenga: está autorizado.
6. Sigue con el siguiente paso **sin preguntar**. Informa en una línea y continúa.

## Paso 3: Cuándo sí detenerse

Solo en estos casos. Conserva el trabajo hecho, explica el bloqueo y di qué decisión hace falta:

- el problema obligaría a cambiar la spec congelada, un umbral, una regla de selección o cualquier
  parámetro experimental ya registrado → se documenta como desviación `X.Y_name.md` y la decide el
  autor;
- aparece un punto de **ASK FIRST** de `CLAUDE.md` (dependencias, caches, modelo de embeddings,
  artefactos versionados, alcance);
- el siguiente paso abre el split de test, cuesta dinero (GPU alquilada, API) o es una ejecución
  única e irrepetible;
- dos intentos seguidos sin progreso en el mismo problema.

Lo demás (un test que falla, un error de tipos, una refactorización local necesaria, un fallo
claro relacionado) se resuelve sin preguntar y se anota para la PR.

## Paso 4: Entregar

Con todos los pasos pedidos completados (modo sin argumento), carga la skill **`deliver`** y
síguela: validación completa, revisión adversarial en contexto limpio, evidencia y PR. La PR queda
esperando la revisión del autor. Nunca se fusiona.

Con un paso o un rango, termina con un resumen corto (pasos hechos, tests, commits) y sugiere
`/6-implementar` o `deliver` cuando el plan esté completo.
