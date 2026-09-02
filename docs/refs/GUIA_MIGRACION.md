# Guía de migración: actualizar framework SDD-WAT

## Resumen de cambios

Esta guía cubre la migración desde la versión anterior del framework (comandos sin numerar, plantillas embebidas en comandos, sin `/scaffold`, `/implementar`, `/verificar` ni `/documentar`) a la versión actual.

---

## 1. Comandos renombrados

Todos los comandos (excepto `/prime`) llevan ahora un prefijo numérico que indica el orden de uso:

| Nombre anterior | Nombre nuevo |
|----------------|--------------|
| `/prime` | `/prime` (sin cambio) |
| `/crea-plan-maestro` | `/1-crea-plan-maestro` |
| — (no existía) | `/2-scaffold` (NUEVO) |
| `/init-project` | `/3-init-project` |
| `/especificar` | `/4-especificar` |
| `/planear` | `/5-planear` |
| — (no existía) | `/6-implementar` (NUEVO) |
| — (no existía) | `/7-verificar` (NUEVO) |
| — (no existía) | `/8-documentar` (NUEVO) |

---

## 2. Acciones a realizar

### Paso 1: Reemplazar `.claude/commands/`

Borrar todos los archivos `.md` de `.claude/commands/` del proyecto destino y copiar los nuevos:

```
Borrar:
  .claude/commands/crea-plan-maestro.md
  .claude/commands/init-project.md
  .claude/commands/especificar.md
  .claude/commands/planear.md
  .claude/commands/prime.md
  .claude/commands/README.md

Copiar desde el framework actualizado:
  .claude/commands/prime.md
  .claude/commands/1-crea-plan-maestro.md
  .claude/commands/2-scaffold.md          ← NUEVO
  .claude/commands/3-init-project.md
  .claude/commands/4-especificar.md
  .claude/commands/5-planear.md
  .claude/commands/6-implementar.md       ← NUEVO
  .claude/commands/7-verificar.md         ← NUEVO
  .claude/commands/8-documentar.md        ← NUEVO
  .claude/commands/README.md
```

### Paso 2: Reemplazar `docs/templates/`

Borrar y copiar completo. Los cambios en las plantillas son:

| Plantilla | Qué cambió |
|-----------|------------|
| `CLAUDE_proyecto.md` | Flujo actualizado con `/6-implementar` y `/7-verificar` |
| `0_plan_maestro.md` | Flujo actualizado con nuevos nombres de comandos |
| `X.spec.md` | Añadida referencia a `/7-verificar` en cabecera |
| `X.0_plan_fase.md` | Pasos incluyen test asociado. Notas sobre ciclo test-first y `/7-verificar` |
| `X.tasks.md` | Sin cambios |
| `X.Y_desviacion.md` | NUEVO — plantilla de desviación con regla de propagación |
| `fix-N_nombre.md` | NUEVO — plantilla de fix con correlativo global y referencia cruzada obligatoria |

**Nota**: los comandos ya NO tienen las plantillas embebidas. Ahora referencian `docs/templates/` por ruta. Si el proyecto no tiene `docs/templates/`, los comandos no podrán generar documentos.

### Paso 3: Reemplazar `CLAUDE_GLOBAL.md`

El CLAUDE.md global (prompt de sistema inyectado en todos los proyectos) ha cambiado significativamente:

- Estructura de archivos incluye los 9 comandos con nombres numerados
- Tabla de comandos ampliada con `/2-scaffold`, `/6-implementar`, `/7-verificar`, `/8-documentar`
- Flujos actualizados
- Sección de testing nueva
- Sección de WAT convertida en opcional
- Referencia a plantillas en `docs/templates/`
- Contenido normalizado a castellano

**Acción**: copiar el `CLAUDE_GLOBAL.md` del framework actualizado y usarlo como prompt de sistema global.

### Paso 4: Crear carpetas si no existen

```bash
mkdir -p docs/templates
mkdir -p docs/plans/fixes
mkdir -p src
mkdir -p tests
mkdir -p data
mkdir -p .tmp
```

### Paso 5: Find-and-replace en documentos del proyecto

Si el proyecto ya tiene documentos generados (specs, planes, CLAUDE.md del proyecto), hay que actualizar las referencias a nombres de comandos antiguos.

**Buscar y reemplazar en todos los `.md` del proyecto** (excepto `docs/refs/`):

| Buscar | Reemplazar por |
|--------|---------------|
| `/crea-plan-maestro` | `/1-crea-plan-maestro` |
| `/init-project` | `/3-init-project` |
| `/especificar` | `/4-especificar` |
| `/planear` | `/5-planear` |

**Importante**: buscar solo cuando van precedidos de `/` (son comandos, no texto suelto). `/prime` no cambia.

Estos son los que existían antes. Los comandos nuevos (`/2-scaffold`, `/6-implementar`, `/7-verificar`, `/8-documentar`) no aparecerán en documentos antiguos, así que no hay nada que reemplazar para ellos.

**Cómo hacerlo rápido con un editor (VS Code):**

1. Abre el proyecto en VS Code
2. `Ctrl+Shift+H` (buscar y reemplazar en archivos)
3. En "files to include": `*.md`
4. En "files to exclude": `docs/refs/**`
5. Ejecuta los 4 reemplazos de la tabla de arriba, uno por uno

**Cómo hacerlo con sed (bash):**

```bash
# Ejecutar desde la raíz del proyecto
find . -name "*.md" -not -path "./docs/refs/*" -not -path "./.git/*" -exec sed -i \
  -e 's|/crea-plan-maestro|/1-crea-plan-maestro|g' \
  -e 's|/init-project|/3-init-project|g' \
  -e 's|/especificar|/4-especificar|g' \
  -e 's|/planear|/5-planear|g' \
  {} +
```

**Nota sobre sed**: este comando es agresivo — reemplaza TODAS las ocurrencias. Revisa el diff después (`git diff`) para asegurarte de que no ha tocado algo inesperado.

### Paso 6: Actualizar el CLAUDE.md del proyecto (si existe)

Si el proyecto ya tiene un `CLAUDE.md` generado, la sección "Plan de evolución" al final probablemente dice:

```
- Fase nueva → /especificar → /planear → implementar
```

Debería decir:

```
- Fase nueva → /4-especificar → /5-planear → /6-implementar → /7-verificar
- Bug fix → /5-planear (si no es trivial) → /6-implementar → /7-verificar
```

---

## 3. Resumen de cambios funcionales

### Comandos nuevos

| Comando | Qué hace |
|---------|----------|
| `/2-scaffold` | Pregunta tipo de proyecto y stack, ejecuta comandos reales del ecosistema (npm create, uv init...), crea la estructura de carpetas del framework |
| `/6-implementar` | Ejecuta los pasos del plan con ciclo test-first (test → red → código → green → marcar [x]). Modos: todo, un paso, rango, siguiente |
| `/7-verificar` | Comprueba alineación spec ↔ código: ejecuta tests, verifica criterios de aceptación, contratos de datos y anti-objetivos |
| `/8-documentar` | Genera guía de usuario desde las specs completadas. Formato breve, con emojis y bullet points, para usuario final no técnico |

### Cambios en comandos existentes

| Comando | Qué cambió |
|---------|------------|
| `/5-planear` | Ya NO ejecuta el trabajo. Solo crea el plan y lo presenta para aprobación. La ejecución va ahora en `/6-implementar` |
| Todos | Las plantillas ya no están embebidas en los comandos. Ahora referencian `docs/templates/` |

### Plantillas nuevas

| Plantilla | Para qué |
|-----------|----------|
| `docs/templates/X.Y_desviacion.md` | Registro de desviación con sección de propagación obligatoria |
| `docs/templates/fix-N_nombre.md` | Corrección de bug con correlativo global y referencia cruzada al plan de fase |

---

## 4. Checklist de migración

- [ ] `.claude/commands/` reemplazado con los 9 comandos nuevos + README
- [ ] `docs/templates/` reemplazado con las 7 plantillas
- [ ] `CLAUDE_GLOBAL.md` (prompt de sistema) reemplazado
- [ ] Carpetas creadas: `docs/templates/`, `docs/plans/fixes/`, `src/`, `tests/`, `data/`, `.tmp/`
- [ ] Find-and-replace de nombres de comandos en documentos existentes del proyecto
- [ ] Sección "Plan de evolución" del CLAUDE.md del proyecto actualizada
- [ ] Verificar con `git diff` que todo está correcto
