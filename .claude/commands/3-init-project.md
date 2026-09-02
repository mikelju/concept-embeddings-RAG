# Comando /3-init-project

Genera o actualiza `CLAUDE.md` **leyendo el código real del proyecto**.

> Este comando requiere que ya exista código. Si el proyecto está vacío, primero ejecuta
> `/1-crea-plan-maestro` para definir el producto, luego crea el scaffold del proyecto
> (ej. `npm create vite@latest`, `npx create-next-app`, etc.) y entonces ejecuta `/3-init-project`.

---

## Por qué CLAUDE.md no se puede escribir a mano desde cero

`CLAUDE.md` no es un documento de intenciones — es un **espejo del código real**.
Contiene los comandos exactos de `package.json`, la arquitectura que realmente existe en los archivos,
los gotchas que solo se descubren leyendo el código, y las convenciones que ya se usan.

Inventarlo antes de que exista el código produce un CLAUDE.md lleno de suposiciones que luego
diverge de la realidad y confunde al agente en lugar de ayudarlo.

**El orden correcto siempre es:**
```
/1-crea-plan-maestro  →  scaffold del proyecto  →  /3-init-project
```

---

## Protocolo

### Paso 1: Verificar que hay código

Comprueba que existe al menos uno de estos archivos:
- `package.json` (proyecto Node/JS/TS)
- `requirements.txt` o `pyproject.toml` (Python)
- `Cargo.toml` (Rust)
- `go.mod` (Go)
- Cualquier archivo de código fuente

Si no existe ninguno, detente y comunica:
> "El proyecto no tiene código todavía. Crea primero el scaffold del proyecto y vuelve a ejecutar `/3-init-project`."

### Paso 2: Leer las fuentes de verdad

Lee en este orden:

1. **`package.json`** (o equivalente) → nombre, versión, scripts, dependencias y devDependencies.
2. **`.env.example`** (si existe) → variables de entorno requeridas.
3. **`docs/plans/0_plan_maestro.md`** (si existe) → descripción del producto y fases.
4. **Archivos de configuración**: `tsconfig.json`, `vite.config.ts`, `next.config.js`, etc.
5. **Archivos principales de código**: puntos de entrada del backend y frontend (inferir por estructura de carpetas).
6. **`README.md`** (si existe) → descripción del proyecto.
7. **`git log --oneline -10`** → historial reciente para entender el estado.

### Paso 3: Inferir la arquitectura

A partir del código leído, determina:

- **Patrón de servidor**: ¿Express puro? ¿Express + Vite middleware? ¿Next.js? ¿API separada del frontend?
- **Gestión de estado/sesión**: ¿cookies? ¿JWT? ¿sesión en memoria? ¿base de datos?
- **Base de datos**: ¿qué ORM o driver? ¿está configurado o solo instalado?
- **Autenticación**: ¿OAuth? ¿JWT? ¿NextAuth? ¿propia?
- **APIs externas**: ¿qué servicios de terceros se usan?
- **Convenciones de código observadas**: nomenclatura, estructura de carpetas, patrones repetidos.

### Paso 4: Identificar gotchas

Los gotchas son las cosas que **solo se saben leyendo el código** y que si el agente no conoce, comete errores.

Busca activamente:
- Soluciones temporales o comentarios que expliquen por qué algo se hace de una manera poco obvia.
- Dependencias instaladas pero no usadas todavía (deuda planificada).
- Variables de entorno con nombres no estándar o lógica de valor por defecto.
- Limitaciones conocidas documentadas en comentarios.
- Configuraciones contraintuitivas (ej. `trust proxy`, `noEmit: true`, etc.).

### Paso 5: Definir permisos del agente y anti-patrones

Este paso define los límites de autonomía del agente y las prácticas prohibidas. Se hace mediante conversación breve con el usuario.

**Pregunta 1** (si no se infiere del código):
> "¿Hay acciones que deba hacer SIEMPRE sin preguntar? Por ejemplo: ejecutar tests, seguir convenciones de nombrado, tipar todo..."

**Pregunta 2:**
> "¿Hay acciones donde deba PARAR y pedirte confirmación antes de ejecutar? Por ejemplo: cambiar esquema de BD, añadir dependencias, tocar CI/CD..."

**Pregunta 3:**
> "¿Hay patrones o prácticas PROHIBIDAS en este proyecto? Por ejemplo: try-catch con rethrow, lógica de negocio en controladores, SQL directo sin ORM..."

Si el usuario no tiene preferencias claras, proponer estos valores por defecto:

```
✅ SIEMPRE: ejecutar tests, seguir convenciones de CLAUDE.md, tipar funciones
⚠️ PREGUNTAR: cambios de esquema de BD, nuevas dependencias, cambios de infra/CI
🚫 NUNCA: commitear secretos, borrar tests sin resolver, editar archivos generados
```

Pedir confirmación antes de incluirlos.

### Paso 6: Proponer tooling de seguridad según el stack

Antes de generar `CLAUDE.md`, proponer al usuario las dependencias de seguridad que complementan al comando `/8-auditar` y al workflow de CI (`.github/workflows/security.yml`). Si el usuario acepta, añadirlas al manifiesto del proyecto como dependencias opcionales.

**Python** (detectado por `pyproject.toml` / `requirements.txt`):

```toml
# Añadir a pyproject.toml -> [project.optional-dependencies] -> security
security = [
    "bandit[toml]>=1.7",       # SAST (patrones inseguros en AST)
    "pip-audit>=2.7",          # CVEs en dependencias
    "detect-secrets>=1.5",     # secretos hardcoded
    "safety>=3.2",             # segunda opinión sobre CVEs
]
```

Instalación: `pip install -e ".[security]"`.

**JS/TS** (detectado por `package.json`):

```json
"devDependencies": {
    "eslint-plugin-security": "^3.0",
    "audit-ci": "^7.1"
}
```

`npm audit` está incluido de fábrica en npm; no requiere instalar nada extra.

**Ambos casos:** mencionar al usuario que el workflow `.github/workflows/security.yml` ya ejecuta estos scanners en CI, así que instalar localmente es opcional para desarrollo (permite `/8-auditar` con escaneo automático en la Phase 2 de la skill).

Si el usuario declina o prefiere posponer, documentarlo en CLAUDE.md en la sección de gotchas ("tooling de seguridad pospuesto — pendiente de instalar").

### Paso 7: Generar CLAUDE.md

Lee la plantilla `docs/templates/CLAUDE_proyecto.md` y úsala como base para generar `CLAUDE.md` en la raíz del proyecto. Rellena cada sección con la información obtenida en los pasos anteriores.

### Paso 8: Confirmación y siguiente paso

Tras escribir el archivo, presenta un resumen de lo que se ha documentado y pregunta:
> "¿Hay algo que haya inferido incorrectamente o que quieras añadir?"

Incorpora las correcciones y confirma que CLAUDE.md está listo.

Indica al usuario:
> "CLAUDE.md generado. A partir de ahora:
> - Ejecuta `/prime` al inicio de cada sesión para cargar el contexto.
> - Ejecuta `/4-especificar` antes de empezar cada fase nueva.
> - Ejecuta `/5-planear` para crear el plan de implementación tras la spec."
