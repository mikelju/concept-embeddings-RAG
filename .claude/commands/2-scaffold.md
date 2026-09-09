# Comando /2-scaffold

Crea la estructura base del proyecto ejecutando los comandos de scaffold del ecosistema elegido.
Se ejecuta DESPUÉS de `/1-crea-plan-maestro` y ANTES de `/3-init-project`.

---

## Cuándo usar este comando

- **Proyecto nuevo**: el plan maestro ya existe pero aún no hay código.
- **Cambio de stack**: se necesita reinicializar el proyecto con otra tecnología.

---

## Protocolo

### Paso 1: Verificar que existe plan maestro

Comprobar si existe `docs/plans/0_plan_maestro.md`.

- Si existe → leerlo para entender qué se va a construir.
- Si NO existe → comunicar:
  > "No hay plan maestro. Ejecuta `/1-crea-plan-maestro` primero para definir las fases del proyecto."

### Paso 2: Preguntar tipo de proyecto

Si el tipo de proyecto no se infiere claramente del plan maestro, preguntar:

> "¿Qué tipo de proyecto es?"
> 1. Backend Python (FastAPI, Flask, Django)
> 2. Frontend web (React, Vue, Svelte, vanilla)
> 3. Fullstack (Next.js, Nuxt, SvelteKit)
> 4. API / microservicio
> 5. CLI / script Python
> 6. Otro (descríbelo)

Esperar respuesta antes de continuar.

### Paso 3: Preguntar detalles del stack

Según el tipo elegido, preguntar los detalles necesarios **de una en una**:

**Si es Python:**
1. "¿Qué framework?" (FastAPI, Flask, Django, ninguno)
2. "¿Gestor de dependencias?" (uv, poetry, pip)
3. "¿Versión de Python?" (3.11, 3.12, 3.13 — sugerir la más reciente estable)
4. "¿Base de datos?" (PostgreSQL, SQLite, MongoDB, ninguna)
5. "¿Framework de testing?" (pytest — sugerir por defecto)

**Si es Node/TypeScript:**
1. "¿Qué framework?" (Express, Next.js, Nuxt, SvelteKit, Vite, otro)
2. "¿Package manager?" (npm, pnpm, bun)
3. "¿TypeScript?" (sí/no — sugerir sí)
4. "¿Base de datos?" (PostgreSQL + Prisma, SQLite, MongoDB, ninguna)
5. "¿Framework de testing?" (Vitest, Jest — sugerir Vitest)

**Si es otro tipo:**
Preguntar qué comandos de scaffold son necesarios y qué estructura espera.

**Preguntas comunes (todos los tipos):**
- "¿Necesitas Docker?" (sí/no)
- "¿Linting/formato?" (sugerir lo estándar del ecosistema: Ruff para Python, ESLint para JS/TS)

Si el usuario no tiene preferencias claras, usar los valores por defecto sugeridos sin preguntar más.

### Paso 4: Ejecutar el scaffold

Ejecutar los comandos estándar del ecosistema. Ejemplos según tipo:

**Python con uv + FastAPI:**
```bash
uv init [nombre-proyecto]
cd [nombre-proyecto]
uv add fastapi uvicorn
uv add --dev pytest ruff mypy
```

**Python con pip + FastAPI:**
```bash
mkdir [nombre-proyecto] && cd [nombre-proyecto]
python -m venv .venv
pip install fastapi uvicorn
pip install -r requirements-dev.txt  # pytest, ruff, mypy
```

**Node con Next.js:**
```bash
npx create-next-app@latest [nombre-proyecto] --typescript --tailwind
```

**Node con Vite + React:**
```bash
npm create vite@latest [nombre-proyecto] -- --template react-ts
cd [nombre-proyecto]
npm install
```

Adaptar los comandos al stack concreto elegido. No inventar — usar los scaffolds oficiales de cada ecosistema.

### Paso 5: Crear estructura de documentación

Después del scaffold, asegurar que existen las carpetas del framework:

```bash
mkdir -p docs/plans/fixes
mkdir -p docs/security
mkdir -p docs/templates
mkdir -p docs/refs
mkdir -p tests
mkdir -p data
mkdir -p .tmp
```

`docs/security/` alberga el catálogo consolidado de hallazgos de `/8-auditar` (`docs/security/README.md`). Se crea ya en el scaffold para que la primera auditoría encuentre el destino listo.

Si `docs/templates/` está vacío y existe una fuente de plantillas (este framework), copiarlas.

### Paso 6: Crear .gitignore si no existe

Si el scaffold no ha generado un `.gitignore`, crear uno adecuado al stack:

- Python: `.venv/`, `__pycache__/`, `.env`, `.tmp/`, etc.
- Node: `node_modules/`, `dist/`, `.env`, `.tmp/`, etc.
- Común: `credentials.json`, `token.json`, `.env.*`, `!.env.example`

### Paso 7: Inicializar git si no existe

```bash
git init  # solo si no existe .git/
```

### Paso 8: Siguiente paso

Comunicar al usuario:
> "Scaffold creado. Ejecuta `/3-init-project` para generar CLAUDE.md leyendo el código real del proyecto."
