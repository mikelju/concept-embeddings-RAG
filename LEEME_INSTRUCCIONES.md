# Framework SDD-WAT

## Qué es esto

Un framework para desarrollar software con agentes de IA (Claude Code) de forma estructurada y predecible. Combina dos sistemas:

- **SDD (Spec-Driven Development)**: antes de escribir código, se define QUÉ construir en una especificación funcional. El agente implementa contra esa spec, no contra instrucciones improvisadas.
- **WAT (Workflows, Agents, Tools)**: opcional, para proyectos con tareas repetibles. Separa las instrucciones (workflows) de la ejecución (scripts Python).

El resultado: en vez de pedirle al agente "hazme un login" y corregir durante horas, le das una spec precisa y el agente construye exactamente lo que necesitas.

---

## Cómo funciona

El framework define un flujo claro para cada fase del proyecto:

```
/4-especificar  →  define QUÉ construir (spec funcional)
       ↓
/5-planear      →  define CÓMO implementarlo (plan de fase)
       ↓
/6-implementar  →  ejecuta el plan paso a paso (ciclo test-first)
       ↓
/7-verificar    →  comprueba que el código cumple la spec
```

Cada paso genera documentación que alimenta al siguiente. La spec define los criterios de aceptación, el plan los convierte en pasos con tests asociados, `/6-implementar` ejecuta cada paso con el ciclo test → falla → código → pasa, y `/7-verificar` comprueba que todo esté alineado.

---

## Los 9 comandos

Se invocan con `/` desde Claude Code. Cada uno tiene un propósito específico:

| Comando | Qué hace | Cuándo usarlo |
|---------|----------|---------------|
| `/prime` | Carga el contexto completo del proyecto | Al inicio de cada sesión |
| `/1-crea-plan-maestro` | Define las fases del producto | Al arrancar un proyecto nuevo |
| `/2-scaffold` | Crea la estructura base del proyecto | Tras crear el plan maestro |
| `/3-init-project` | Genera CLAUDE.md leyendo el código real | Tras crear el scaffold del proyecto |
| `/4-especificar` | Genera la spec funcional de una fase | Antes de empezar cada fase |
| `/5-planear` | Genera el plan de implementación | Después de la spec, antes de codificar |
| `/6-implementar` | Ejecuta los pasos del plan (test-first) | Después del plan aprobado |
| `/7-verificar` | Comprueba alineación spec ↔ código | Al terminar una fase |
| `/8-documentar` | Genera guía de usuario desde las specs | Al terminar una fase o el proyecto |

Los protocolos completos están en `.claude/commands/`. La referencia rápida en `.claude/commands/README.md`.


---

## Las plantillas

Cada tipo de documento tiene una plantilla en `docs/templates/`. Los comandos las leen cuando generan documentos nuevos. Si quieres cambiar el formato de un documento, tocas la plantilla y todos los comandos usan la versión nueva.

| Plantilla | Qué genera | Qué comando la usa |
|-----------|------------|-------------------|
| `CLAUDE_proyecto.md` | CLAUDE.md del proyecto | `/3-init-project` |
| `0_plan_maestro.md` | Plan maestro con fases | `/1-crea-plan-maestro` |
| `X.spec.md` | Spec funcional de una fase | `/4-especificar` |
| `X.0_plan_fase.md` | Plan de implementación | `/5-planear` |
| `X.tasks.md` | Tareas atómicas (opcional) | `/5-planear` |
| `X.Y_desviacion.md` | Registro de desviación | `/6-implementar` |
| `fix-N_nombre.md` | Corrección de bug puntual | Manual |

---

## Estructura de archivos

Así se ve un proyecto que usa este framework:

```
mi-proyecto/
├── CLAUDE.md                  # Constitución del proyecto (generado por /3-init-project)
├── .claude/commands/          # Los 9 comandos + README
├── src/                       # Código fuente
├── tests/                     # Tests automatizados
├── docs/
│   ├── plans/                 # Documentación SDD + planes
│   │   ├── 0_plan_maestro.md  # Visión global — fases y estado
│   │   ├── fase_1/
│   │   │   ├── 1.spec.md     # Spec funcional (QUÉ)
│   │   │   ├── 1.0_nombre.md # Plan de fase (CÓMO)
│   │   │   └── 1.Y_*.md      # Desviaciones (si las hay)
│   │   └── fixes/
│   │       └── fix-N_*.md    # Bug fixes puntuales
│   └── templates/             # Plantillas de documentos
├── .env                       # Variables de entorno
└── .gitignore
```

---

## Flujo completo de un proyecto

### Arranque (una vez)

```
1. /1-crea-plan-maestro    →  define qué construyes y en qué fases
2. /2-scaffold             →  inicializar el proyecto (pip init, npm create, etc.)
3. /3-init-project         →  genera CLAUDE.md desde el código real
```

### Cada fase (repetir)

```
4. /4-especificar          →  spec funcional: QUÉ + POR QUÉ
5. /5-planear              →  plan de implementación: CÓMO (usuario revisa y aprueba)
6. /6-implementar          →  código + tests paso a paso (test-first)
7. /7-verificar            →  comprobar alineación spec ↔ código → marcar [x]
8. /8-documentar           →  guía de usuario (opcional, al terminar fase o proyecto)
```

### Cada sesión de trabajo

```
/prime                     →  cargar contexto antes de empezar
```

---

## Cómo desplegar este framework en un proyecto nuevo

### Paso 1: Prompt de sistema global
Copia el contenido de `CLAUDE_GLOBAL.md` a tu CLAUDE.md global (el que se inyecta como prompt de sistema en todos los proyectos).

### Paso 2: Comandos
Copia la carpeta `.claude/commands/` completa a tu sistema. Los comandos se invocan con `/` desde cualquier proyecto.

### Paso 3: Plantillas
Copia la carpeta `docs/templates/` al directorio `docs/templates/` de cada proyecto nuevo. Los comandos las leen desde esa ruta cuando generan documentos.

### Paso 4: Verificar
1. Ejecuta `/1-crea-plan-maestro` → debería generar el plan maestro desde la plantilla
2. Ejecuta `/2-scaffold` → debería crear la estructura base del proyecto
3. Ejecuta `/3-init-project` → debería generar CLAUDE.md
4. Ejecuta `/4-especificar` → `/5-planear` → `/6-implementar` → `/7-verificar`

---

## Framework WAT (opcional)

WAT solo se necesita si el proyecto tiene **tareas repetibles** (scraping, reporting, pipelines de datos). Para aplicaciones web o APIs estándar, basta con SDD + planes.

Si se usa WAT, se añaden estas carpetas:

```
workflows/          # Instrucciones de proceso (Markdown)
tools/              # Scripts de ejecución (Python)
docs/learnings/     # Lecciones aprendidas por workflow
```

Los detalles completos de WAT están en `CLAUDE_GLOBAL.md`, sección "Framework WAT (opcional)".

---

## Documentos de referencia

En `docs/refs/` hay dos documentos que explican la teoría detrás del framework:

- **SDD_Guia_Integrada_Definitiva.md** — Guía general de Spec-Driven Development: qué es, niveles de compromiso, estructura de specs, errores comunes.
- **SDD_Integracion_WAT_Planes.md** — Cómo se integra SDD con el framework WAT y el sistema de planes. Incluye ejemplo práctico.

Son documentos de consulta, no operativos. Los comandos y plantillas son los que se usan en el día a día.

---

## Qué hay en este paquete

```
├── CLAUDE_GLOBAL.md              # Prompt de sistema para el agente
├── LEEME_INSTRUCCIONES.md        # Este archivo
├── .claude/
│   └── commands/                 # 9 comandos + README
├── docs/
│   ├── templates/                # 7 plantillas de documentos
│   └── refs/                     # 2 documentos de referencia teórica
```
