# Comando /7-verificar

Verifica que el código implementado está alineado con la especificación funcional de la fase.
Se ejecuta al terminar una fase o a mitad de fase para comprobar el progreso.

---

## Cuándo usar este comando

- **Al terminar una fase**: antes de marcar la fase como completada en el plan maestro.
- **A mitad de fase**: para comprobar el progreso y detectar deriva temprana.
- **Después de un fix o desviación**: para verificar que no se ha roto la alineación.

---

## Protocolo

### Paso 1: Cargar contexto

Lee en este orden:

1. **`CLAUDE.md`** → obtener el comando de tests del proyecto (sección "Comandos de desarrollo").
2. **`docs/plans/fase_X/X.spec.md`** → criterios de aceptación, contratos de datos, anti-objetivos.
3. **`docs/plans/fase_X/X.0_nombre.md`** → estado del plan, archivos afectados.

Si no existe spec para la fase → comunicar:
> "No hay spec para esta fase. No se puede verificar alineación. Ejecuta `/4-especificar` primero."

Si el usuario no indica fase → verificar la fase activa (la primera no completada).

### Paso 2: Ejecutar tests

Ejecutar el comando de tests del proyecto (definido en `CLAUDE.md`).

Registrar:
- Qué tests pasan
- Cuáles fallan (con mensaje de error)
- Si hay criterios de aceptación sin test asociado

Si `CLAUDE.md` no define un comando de tests → preguntar al usuario cómo ejecutarlos.

### Paso 3: Verificar criterios de aceptación

Para CADA criterio de aceptación de la spec:

1. **¿Existe al menos un test que lo cubra?**
   - Buscar en la carpeta de tests un test que verifique este criterio.
   - Si no existe → marcar como `[ ] Sin test`.

2. **¿El test pasa?**
   - Si pasa → marcar como `[x]`.
   - Si falla → marcar como `[!] Test falla`.

### Paso 4: Verificar contratos de datos

Si la spec define contratos de datos (modelos, tablas, endpoints):

1. **Modelos/Tablas**: ¿Los campos, tipos y restricciones del código coinciden con los de la spec?
2. **Endpoints**: ¿Las rutas, métodos, request y response del código coinciden con los de la spec?
3. Si hay divergencia → anotar qué difiere y en qué archivo.

### Paso 5: Verificar anti-objetivos

Revisar la sección "Anti-objetivos" de la spec. ¿Se ha implementado algo que la spec excluía explícitamente? Si sí → anotar.

### Paso 6: Generar informe

Presentar al usuario:

```
## Verificación: Fase X — [Nombre]

### Criterios de aceptación
- [x] [criterio 1] — test: `tests/ruta/test_archivo::test_nombre`
- [!] [criterio 2] — test falla: [error resumido]
- [ ] [criterio 3] — sin test

### Contratos de datos
- [x] Modelo `tabla` — campos coinciden con spec
- [!] Endpoint `POST /ruta` — response difiere: spec dice X, código devuelve Y

### Anti-objetivos
- [x] No se implementó [cosa excluida]
- [!] Se implementó [cosa que la spec excluía]

### Resultado
[X de Y criterios verificados] | [N pasan, M fallan, K sin test]

### Acciones pendientes
- [crear test para criterio 3]
- [corregir response de POST /ruta]
- ...
```

### Paso 7: Resolver divergencias

Si hay drift (divergencias entre spec y código), preguntar al usuario:

> "Hay divergencias entre la spec y el código. ¿Actualizamos la spec para reflejar el código actual, o corregimos el código para cumplir la spec?"

Aplicar la decisión del usuario. Si se actualiza la spec → propagar cambios al plan de fase.

### Paso 8: Marcar completado

Si todos los criterios de aceptación pasan:
- Marcar criterios como `[x]` en la spec.
- Marcar pasos correspondientes como `[x]` en el plan de fase.
- **NO marcar la fase como cerrada en el plan maestro todavía**: la fase se cierra al completar `/8-auditar` + `/9-documentar`.
- Comunicar:
  > "Fase X verificada contra la spec. Siguiente paso: `/8-auditar` para la auditoría de seguridad.
  > Si la fase no toca código de seguridad (solo docs/assets/refactor sin cambio de comportamiento), el propio `/8-auditar` ofrecerá el escape hatch."

### Flujo después de verificar

La verificación funcional **no cierra la fase**. El flujo completo es:

```
/7-verificar → /8-auditar → /9-documentar → marcar fase en plan maestro
```

Ver `CLAUDE.md` / `CLAUDE_GLOBAL.md` → sección "Auditoría de seguridad" para las reglas de cierre de fase tras la auditoría.
