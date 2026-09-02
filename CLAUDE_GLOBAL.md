# Instrucciones para el agente

Este documento define cómo debe operar el agente en cualquier proyecto. Contiene la estructura de archivos, los comandos disponibles, el sistema de documentación, las reglas de testing, las reglas de codificación y, opcionalmente, el framework WAT para proyectos que lo necesiten.

**Leer siempre este documento completo antes de empezar a trabajar en cualquier proyecto.**

---

## Estructura de archivos del proyecto

Todo proyecto sigue esta estructura base. No todos los proyectos usan todas las carpetas — las marcadas como opcionales se crean solo cuando son necesarias.

```
mi-proyecto/
├── CLAUDE.md                          # Constitución del proyecto — leer siempre antes de actuar
├── .claude/
│   ├── commands/                      # Comandos transversales (compartidos entre proyectos)
│   │   ├── prime.md
│   │   ├── 1-crea-plan-maestro.md
│   │   ├── 2-scaffold.md
│   │   ├── 3-init-project.md
│   │   ├── 4-especificar.md
│   │   ├── 5-planear.md
│   │   ├── 6-implementar.md
│   │   ├── 7-verificar.md
│   │   ├── 8-auditar.md
│   │   └── 9-documentar.md
│   ├── skills/                        # Skills reutilizables (SKILL.md + references/)
│   │   └── audit-code/                # Auditoría de seguridad profesional (Python + React)
│   ├── settings.local.json
│   └── README.md
├── src/                               # Código fuente de la aplicación
├── tests/                             # Tests automatizados
├── data/                              # Archivos de datos temporales (CSV, Excel, JSON...)
├── docs/
│   ├── plans/                         # Sistema de documentación SDD + planes
│   │   ├── 0_plan_maestro.md          # Visión global — fases, estado, decisiones
│   │   ├── fase_1/
│   │   │   ├── 1.spec.md             # Especificación funcional (QUÉ + POR QUÉ)
│   │   │   ├── 1.0_nombre_fase.md    # Plan de implementación (CÓMO)
│   │   │   ├── 1.tasks.md            # (Opcional) Tareas atómicas — solo fases complejas
│   │   │   └── 1.Y_desviacion.md     # Desviación o ajuste fuera del plan (correlativo)
│   │   └── fixes/
│   │       └── fix-N_nombre.md       # Bug fix puntual (correlativo global)
│   ├── security/                      # Informes de auditoría de seguridad (/8-auditar)
│   │   ├── README.md                  # Catálogo de hallazgos SEC/DEF/OBS
│   │   └── audit-YYYY-MM-DD-*.md      # Informes por fase / release gate
│   ├── templates/                     # Plantillas de documentos (spec, plan, tasks, CLAUDE.md...)
│   ├── refs/                          # Documentación de referencia inmutable (PDFs, manuales, etc.)
│   └── learnings/                     # (Solo con WAT) Lecciones por workflow
├── output/                            # (Opcional) Entregables generados
├── workflows/                         # (Solo con WAT) SOPs específicos del proyecto
├── tools/                             # (Solo con WAT) Scripts de ejecución determinista
├── .tmp/                              # Archivos temporales — regenerables, desechables
├── .env                               # Claves API y variables de entorno (NUNCA en otro sitio)
└── .gitignore
```

---

## Comandos transversales (.claude/commands/)

Los comandos transversales son herramientas reutilizables que funcionan en todos los proyectos. Residen en `.claude/commands/` y se invocan con `/`. Gestionan la documentación, la planificación y el contexto del proyecto.

| Comando | Cuándo usarlo | Qué genera |
|---------|---------------|------------|
| `/prime` | Al inicio de cada sesión | Informe de contexto del proyecto |
| `/1-crea-plan-maestro` | Al crear un proyecto nuevo | `docs/plans/0_plan_maestro.md` |
| `/2-scaffold` | Tras crear el plan maestro | Estructura base del proyecto |
| `/3-init-project` | Tras crear el scaffold del proyecto | `CLAUDE.md` (leyendo código real) |
| `/4-especificar` | Antes de empezar cada fase nueva | `docs/plans/fase_X/X.spec.md` |
| `/5-planear` | Después de la spec, antes de codificar | `docs/plans/fase_X/X.0_nombre.md` |
| `/6-implementar` | Después del plan, para ejecutar los pasos | Código + tests (ciclo test-first) |
| `/7-verificar` | Al terminar una fase (o a mitad para comprobar) | Informe de alineación spec ↔ código |
| `/8-auditar` | Antes de cerrar una fase, antes de release, tras un fix crítico | `docs/security/audit-YYYY-MM-DD-<modo>.md` |
| `/9-documentar` | Al terminar una fase o el proyecto | `docs/GUIA_USUARIO.md` |

### Flujo de un proyecto nuevo

```
/1-crea-plan-maestro → /2-scaffold → /3-init-project
```

### Flujo de cada fase

```
/4-especificar → /5-planear → /6-implementar → /7-verificar → /8-auditar → /9-documentar → marcar [x]
```

### Release gate (antes de empaquetar / desplegar / publicar)

```
/8-auditar completo   → auditoría full-project para cazar issues cruzados
```

### Flujo de cada sesión de trabajo

```
/prime → (cargar contexto) → trabajar
```

---

## Sistema de documentación del proyecto (SDD + Planes)

El proyecto utiliza **Spec-Driven Development (SDD)** integrado con un sistema de planes jerárquico. La combinación asegura que cada fase tiene una definición precisa de QUÉ construir (spec), un plan de CÓMO implementarlo (plan de fase) y un registro de todo lo que ocurre durante la ejecución (desviaciones, fixes).

### Tipos de documento

| Archivo | Qué contiene | Cuándo se crea | Quién lo crea |
|---------|-------------|----------------|---------------|
| `0_plan_maestro.md` | Visión global, fases, estado | Al inicio del proyecto | `/1-crea-plan-maestro` |
| `X.spec.md` | Especificación funcional: QUÉ + POR QUÉ | Antes de cada fase | `/4-especificar` |
| `X.0_nombre.md` | Plan de implementación: CÓMO | Después de la spec | `/5-planear` |
| `X.tasks.md` | Tareas atómicas (opcional) | Solo si la fase tiene >10 pasos | `/5-planear` o manual |
| `X.Y_nombre.md` | Desviación o ajuste fuera del plan | Cuando aparece un problema | Manual |
| `fix-N_nombre.md` | Corrección de bug puntual | Cuando se detecta un bug | Manual |

### Plantillas

Las plantillas de cada tipo de documento están en `docs/templates/`. Los comandos (`/4-especificar`, `/5-planear`, `/1-crea-plan-maestro`, `/3-init-project`) las usan como referencia al generar documentos nuevos.

| Plantilla | Para qué |
|-----------|----------|
| `CLAUDE_proyecto.md` | Plantilla del CLAUDE.md de cada proyecto |
| `0_plan_maestro.md` | Plantilla del plan maestro |
| `X.spec.md` | Plantilla de especificación funcional |
| `X.0_plan_fase.md` | Plantilla del plan de fase |
| `X.tasks.md` | Plantilla de tareas atómicas (opcional) |
| `X.Y_desviacion.md` | Plantilla de desviación o ajuste fuera del plan |
| `fix-N_nombre.md` | Plantilla de corrección de bug puntual |

### Regla de oro

**Antes de tocar código, hay una spec y un plan documentados.** Si no hay spec, se ejecuta `/4-especificar`. Si no hay plan, se ejecuta `/5-planear`.

### Regla obligatoria: especificar y planificar antes de codificar

**Antes de implementar cualquier cambio, sigue este orden:**

1. **Si es una fase nueva** → ejecuta `/4-especificar` para definir QUÉ se construye
2. **Luego** → ejecuta `/5-planear` para documentar CÓMO se construye
3. **No toques código** (ni crees archivos, ni edites nada) hasta que spec + plan estén escritos y el usuario los confirme

**Excepciones que NO requieren ni spec ni plan:**
- Arreglos de una línea (typos, imports rotos)
- Ejecución de workflows o tools existentes sin modificar código

**Excepciones que requieren plan pero NO spec:**
- Bug fixes que requieran más de un cambio trivial (van a `fix-N`)
- Cambios que afectan a 1-2 archivos sin alterar comportamiento funcional

**Excepciones que requieren spec pero plan mínimo:**
- Cambios pequeños que alteran comportamiento funcional (nuevos campos, nuevo endpoint) — spec rápida con objetivo + criterios de aceptación

### Reglas de actualización

- **Spec**: NO se modifica durante la implementación. Si el alcance cambia → actualizar spec primero, luego propagar al plan.
- **Plan maestro**: Solo se actualiza tras verificar que las cosas funcionan, no al planificar.
- **Plan de fase**: Se mantiene siempre como reflejo fiel del estado real. Si una desviación cambia el enfoque → actualizar.
- **Propagación**: Todo fluye hacia arriba — desviaciones actualizan el plan de fase, cambios sustanciales en el plan de fase se reflejan en el plan maestro.

### Regla de spec en código

Todo proceso (manual o automatizado) que produzca o modifique código debe:
1. Leer `CLAUDE.md` (constitución del proyecto)
2. Leer la spec de la fase activa (`docs/plans/fase_X/X.spec.md`) si existe
3. Verificar que el resultado cumple los criterios de aceptación de la spec

Si no existe spec para la fase activa, ejecutar `/4-especificar` primero.

---

## Testing y verificación contra especificaciones

Estas reglas aplican a todos los proyectos, independientemente del stack o del framework de testing que se use.

### Reglas generales de testing

1. **Todo criterio de aceptación de la spec debe tener al menos un test asociado.** Si la spec dice "no se pueden reservar dos citas en el mismo hueco", debe existir un test que lo verifique.

2. **Los tests se escriben ANTES o AL MISMO TIEMPO que la implementación**, nunca después. Si el plan de fase tiene un paso "Crear endpoint POST /api/citas", el paso debe incluir la creación del test correspondiente.

3. **No se puede marcar una tarea como completada `[x]` sin que sus tests pasen.** Si los tests no pasan, la tarea sigue pendiente.

4. **Nunca eliminar tests que estén fallando sin resolver el problema.** Si un test falla, se arregla el código o se documenta por qué el test ya no es válido (con actualización de la spec si corresponde).

5. **Al terminar una fase, verificar los criterios de aceptación de la spec uno por uno.** Cada criterio se marca como `[x]` solo si existe un test que lo valida y ese test pasa.

### Verificación de alineación spec ↔ código

Antes de dar una fase por completada, comprobar:

- ¿Todos los criterios de aceptación de la spec tienen tests? → Si falta alguno, crearlo.
- ¿Los contratos de datos (modelos, endpoints) del código coinciden con los de la spec? → Si divergen, actualizar la spec o corregir el código.
- ¿Hay funcionalidad implementada que no está en la spec? → Si es útil, añadirla a la spec. Si no lo es, eliminarla del código.
- ¿Los anti-objetivos de la spec se respetan? → Verificar que no se ha implementado nada que la spec excluía explícitamente.

### Framework de testing

El framework de testing concreto (Jest, Vitest, Pytest, etc.) se define en el `CLAUDE.md` de cada proyecto, junto con los comandos de ejecución. Las reglas de esta sección aplican independientemente del framework elegido.

---

## Auditoría de seguridad

Todo proyecto integra una auditoría de seguridad estructurada como paso del flujo de fase. Se apoya en la skill `audit-code` (`.claude/skills/audit-code/`) que contiene el protocolo, el catálogo de vulnerabilidades (Python, React, OWASP Top 10, patrones de secretos) y la plantilla de informe.

### Cuándo se audita

- **Obligatorio al cerrar una fase** que toque código de seguridad (autenticación, criptografía, entrada de usuario, dependencias externas, manejo de ficheros, subprocess, deserialización).
- **Obligatorio antes de cualquier release** (modo `completo` — release gate).
- **Recomendado tras un fix** en módulos sensibles.
- **Ad-hoc** cuando se pida una revisión de seguridad.

### Escape hatch

Si la fase activa no toca código de seguridad relevante (solo docs, assets o refactor puro sin cambio de comportamiento), `/8-auditar` pregunta si saltar la auditoría y registra la exención en `docs/security/` para trazabilidad. Nunca saltar fases que toquen auth, crypto, input externo, subprocess o deserialización.

### Rúbrica de severidad

Cinco niveles consistentes con el catálogo de la skill:

| Severidad | Qué significa en la práctica |
|-----------|------------------------------|
| Critical | Explotable sin condiciones; parar y arreglar hoy. |
| High | Explotable con condiciones realistas; fix antes de release. |
| Medium | Explotable con condiciones específicas, o con impacto acotado. Fix en el sprint. |
| Low | Defensa en profundidad, no explotable por sí solo. Backlog. |
| Info | Observación, no hay riesgo directo. |

Cada hallazgo recibe un ID `SEC-NNN` (la centena indica la fase: `1xx`=Fase 1, `3xx`=Fase 3, etc.), un tag CWE y una categoría OWASP.

### Cierre de fase tras auditoría

- Si no hay hallazgos **Critical ni High** → la fase puede cerrarse. Pasar a `/9-documentar`.
- Si hay Critical o High → la fase **no se cierra** hasta resolverlos o registrar decisión explícita de aplazar con justificación en el plan maestro.
- Cada SEC cerrado se resuelve con un `fix-N` en `docs/plans/fixes/` siguiendo el protocolo del proyecto.

### Catálogo centralizado

Cada proyecto mantiene `docs/security/README.md` como índice único de hallazgos (pendientes, cerrados, aplazados por decisión de negocio). Este archivo es el punto de entrada para revisar el estado de seguridad sin tener que abrir los 6 informes por fase.

### Automatización en CI

La plantilla incluye `.github/workflows/security.yml`. Detecta automáticamente el stack (Python / Node) y ejecuta **bandit**, **pip-audit**, **detect-secrets** o **npm audit** según corresponda, en cada push y pull request sobre `main`/`master`.

Funciona como **red de seguridad complementaria a `/8-auditar`**: si un desarrollador olvida ejecutar la auditoría manual, el CI cazará como mínimo las vulnerabilidades que los escáneres automáticos detectan (SAST + dependencias + secretos). `/8-auditar` sigue siendo necesario para la revisión manual y el razonamiento de explotabilidad — el CI no sustituye el análisis humano, lo refuerza.

Si un hallazgo del CI es falso positivo, documentar la excepción en `docs/security/` en lugar de suprimirla silenciosamente.

### Reglas no negociables durante la auditoría

1. Nunca instalar herramientas (`pip install`, `npm install`) sin autorización explícita.
2. Nunca ejecutar payloads contra el propio código ni sistemas externos — el PoC es descriptivo.
3. Nunca volcar secretos completos en el informe — redactar siempre (`sk-abcd•••`).
4. Nunca decir "el código es seguro" — usar "sin hallazgos críticos bajo el alcance revisado".
5. Falsos positivos > falsos negativos. Ante la duda, reportar como HIGH y dejar al humano degradar.
6. No modificar código durante la auditoría — solo tras el informe y con luz verde explícita.

---

## Skills reutilizables

Las skills en `.claude/skills/` encapsulan protocolos especializados que se disparan automáticamente cuando su descripción matchea la intención del usuario (o explícitamente con `/<skill-name>`). Residen en el proyecto (versionadas) para que todo el equipo las comparta.

Estructura de una skill:

```
.claude/skills/<nombre>/
├── SKILL.md              # Frontmatter con name + description + triggers + contenido principal
└── references/           # (Opcional) archivos auxiliares cargados bajo demanda
```

Skills incluidas por defecto en la plantilla:

| Skill | Para qué |
|-------|----------|
| `audit-code` | Auditoría de seguridad profesional (ver sección anterior) |

Cuando una skill crece, mantener el SKILL.md corto y mover los catálogos a `references/*.md` cargados solo cuando el flujo los necesita.

---

## Reglas de codificación

### Python: caracteres Unicode

**No usar caracteres Unicode especiales en código Python.** El backend se ejecuta en Docker (Gunicorn) y en Google Cloud Run, donde `stdout` puede usar codecs como `charmap` que no soportan caracteres fuera de ASCII extendido. Esto causa errores de tipo `'charmap' codec can't encode character`.

**Prohibido en print(), logging, strings de código:**
- Emojis: ✓ ✗ ✔ ✘ 💻 👉 📜 🚀 🏗️ y similares
- Flechas Unicode: → ← ↑ ↓
- Símbolos especiales: • ■ □ ★

**Usar en su lugar:**
- `[OK]` en vez de `✓`
- `[ERROR]` en vez de `✗`
- `[WARN]` en vez de `⚠`
- `->` en vez de `→`
- `-` o `*` en vez de `•`

**Excepción:** Los archivos `.md` de documentación SÍ pueden usar estos caracteres porque no pasan por stdout de Python.

### Convenciones específicas del proyecto

Las convenciones de nombrado, formato, patrones y anti-patrones específicos se definen en el `CLAUDE.md` de cada proyecto (generado por `/3-init-project`). Incluyen:

- Estilo de código (con snippets de ejemplo)
- Permisos del agente (SIEMPRE / PREGUNTAR / NUNCA)
- Anti-patrones prohibidos (con alternativas)
- Gotchas conocidos

---

## Framework WAT (opcional)

El framework **WAT** (Workflows, Agents, Tools) es un sistema para separar la instrucción del trabajo (workflows) de la ejecución determinista (tools). **No todos los proyectos lo necesitan.** Se activa cuando el proyecto tiene tareas repetibles que se benefician de scripts reutilizables (scraping, reporting, pipelines de datos, integraciones con APIs externas, etc.).

### Cuándo usar WAT

- **SÍ usar** cuando el proyecto tiene procesos repetibles que se ejecutan muchas veces (ej. scraping semanal, generación de informes, sincronización de datos).
- **NO usar** cuando el proyecto es una aplicación web o API estándar donde el código ES el producto. En ese caso, basta con el sistema de documentación SDD + planes.

### Cómo funciona

**Capa 1: Workflows (las instrucciones)**
- Archivos Markdown en `workflows/` que describen procesos paso a paso.
- Cada workflow define: objetivo, inputs necesarios, qué tools usar, outputs esperados y cómo manejar errores.
- Son específicos de cada proyecto: un workflow de scraping, uno de reporting, uno de procesamiento de datos...

**Capa 2: Agente (el coordinador)**
- Tu rol como agente. Lees el workflow, ejecutas los tools en el orden correcto, gestionas errores y pides aclaraciones cuando hace falta.
- Conectas la intención con la ejecución sin intentar hacerlo todo tú directamente.
- Ejemplo: si necesitas datos de una web, no lo intentes directamente. Lee `workflows/scrape_website.md`, identifica los inputs necesarios, y ejecuta `tools/scrape_single_site.py`.

**Capa 3: Tools (la ejecución)**
- Scripts Python en `tools/` que hacen el trabajo real.
- Llamadas a APIs, transformaciones de datos, operaciones de archivos, consultas a base de datos.
- Las credenciales y claves API están en `.env`.
- Son consistentes, testeables y rápidos.

**¿Por qué importa esta separación?** Cuando la IA intenta manejar cada paso directamente, la precisión cae rápido. Si cada paso tiene un 90% de acierto, tras cinco pasos estás en el 59%. Al delegar la ejecución a scripts deterministas, el agente se concentra en lo que hace bien: coordinación y toma de decisiones.

### Carpetas de WAT

Si el proyecto usa WAT, estas carpetas se añaden a la estructura base:

```
workflows/          # SOPs del proyecto (Markdown) — instrucciones de proceso
tools/              # Scripts de ejecución (Python) — acción determinista
docs/learnings/     # Lecciones aprendidas por workflow
```

**`docs/learnings/`** captura qué funcionó y qué no en cada ejecución de workflow. Al terminar un workflow, se revisa con el usuario y se actualiza el archivo de learnings correspondiente. Es la memoria institucional que hace que cada ejecución sea mejor que la anterior.

### Reglas de operación con WAT

**1. Buscar tools existentes primero.**
Antes de construir algo nuevo, comprobar `tools/` según lo que requiera el workflow. Solo crear scripts nuevos cuando no exista nada para esa tarea.

**2. Aprender y adaptarse cuando algo falle.**
Cuando ocurra un error:
- Leer el mensaje de error completo y el trace.
- Arreglar el script y volver a probar (si usa llamadas a APIs de pago, consultar con el usuario antes de ejecutar de nuevo).
- Documentar lo aprendido en el workflow (rate limits, timing, comportamiento inesperado).

**3. Mantener los workflows actualizados.**
Los workflows deben evolucionar conforme se aprende. Cuando se encuentren mejores métodos, restricciones o problemas recurrentes, actualizar el workflow. Eso sí, no crear ni sobreescribir workflows sin preguntar al usuario salvo que se indique explícitamente.

### Bucle de auto-mejora

Cada fallo es una oportunidad para fortalecer el sistema:
1. Identificar qué se rompió
2. Arreglar el tool o workflow
3. Verificar que la corrección funciona
4. Actualizar el workflow con el nuevo enfoque
5. Capturar la lección en `docs/learnings/`
6. Continuar con un sistema más robusto

---

## Resumen de comportamiento

Estás entre lo que el usuario quiere (documentado en specs, planes y workflows) y lo que realmente se ejecuta (código y tools). Tu trabajo es leer las instrucciones, tomar decisiones inteligentes, ejecutar o delegar en los tools correctos, recuperarte de errores, verificar contra las specs, y mejorar el sistema continuamente.

Sé pragmático. Sé fiable. Sigue aprendiendo.
