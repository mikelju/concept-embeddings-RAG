# CLAUDE.md — [Nombre del Proyecto]

Reglas y contexto del proyecto para el agente. Leer siempre antes de hacer cambios.

---

## ¿Qué es este proyecto?

[2-3 frases: qué hace, para quién, estado actual del desarrollo]

---

## Comandos de desarrollo

```bash
[comando-dev]       # Arrancar en modo desarrollo
[comando-test]      # Ejecutar tests
[comando-build]     # Compilar para producción
[comando-lint]      # Linting y formato
[comando-db]        # Migraciones de base de datos (si aplica)
```

**Para arrancar:** `[comando exacto]` — abre `[URL]`

---

## Arquitectura

### Estructura de archivos

```
src/
├── [carpeta]/      # [descripción de qué contiene]
├── [carpeta]/      # [descripción]
├── [carpeta]/      # [descripción]
└── [archivo]       # [descripción]
tests/
└── [estructura]    # [espejo de src/ o convención usada]
```

### [Patrón arquitectónico principal]

[Explicación del patrón que sigue el proyecto (MVC, Router→Service→DataClient,
capas hexagonales, etc.) y por qué se eligió así]

### [Otros patrones relevantes]

[Otros patrones importantes: cómo se gestiona el estado, cómo se manejan
las conexiones a BD, cómo se estructuran las rutas, etc.]

---

## Tech Stack

| Capa | Tecnología | Versión |
|------|-----------|---------|
| Runtime | [ej. Node.js] | [ej. 20 LTS] |
| Framework | [ej. Express] | [ej. 4.18] |
| Lenguaje | [ej. TypeScript] | [ej. 5.3] |
| Base de datos | [ej. PostgreSQL + Prisma] | [ej. 16 / 5.x] |
| Tests | [ej. Vitest] | [ej. 1.x] |
| Despliegue | [ej. Docker + Cloud Run] | — |

---

## Variables de entorno

```env
DATABASE_URL=        # Conexión a la base de datos
[VARIABLE]=          # [para qué se usa]
[VARIABLE]=          # [para qué se usa]
```

---

## Convenciones de código

- [Convención observada en el código real — ej. "Funciones async/await, nunca callbacks"]
- [Convención — ej. "Exportaciones nombradas, nunca default export"]
- [Convención — ej. "camelCase para variables, PascalCase para componentes"]
- [Convención — ej. "Tests colocados en tests/ con estructura espejo de src/"]

---

## Permisos del agente

✅ **SIEMPRE** (hacer sin preguntar):
- [ej. Ejecutar tests antes de dar por completada una tarea]
- [ej. Seguir las convenciones de nombrado del proyecto]
- [ej. Tipar todas las funciones nuevas]

⚠️ **PREGUNTAR PRIMERO** (suspender y pedir confirmación):
- [ej. Modificar el esquema de base de datos]
- [ej. Añadir nuevas dependencias a package.json]
- [ej. Cambiar configuración de CI/CD o despliegue]

🚫 **NUNCA** (prohibido sin excepción):
- [ej. Commitear secretos o credenciales]
- [ej. Borrar tests sin resolverlos primero]
- [ej. Editar archivos en node_modules/ o archivos generados]
- [ej. Escribir lógica de negocio directamente en controladores/routers]

---

## Anti-patrones del proyecto

🚫 [Patrón prohibido] — [por qué está prohibido y qué hacer en su lugar]
🚫 [Patrón prohibido] — [alternativa correcta]
🚫 [Patrón prohibido] — [alternativa correcta]

---

## Gotchas conocidos

1. [Cosa no obvia que si el agente no sabe, cometerá un error]
2. [Configuración contraintuitiva y por qué existe]
3. [Solución temporal activa y cuándo se puede eliminar]
4. [Limitación conocida que no se puede resolver ahora]

---

## Plan de evolución

El roadmap completo está en `docs/plans/0_plan_maestro.md`.

**Antes de cualquier cambio:**
- Fase nueva → `/4-especificar` → `/5-planear` → `/6-implementar` → `/7-verificar` → `/8-auditar` → `/9-documentar`
- Bug fix → `/5-planear` (si no es trivial) → `/6-implementar` → `/7-verificar` → `/8-auditar` (si toca auth, crypto, input externo, subprocess o deserialización)
- Antes de empaquetar/release → `/8-auditar completo` como release gate

Invocar `/prime` al inicio de cada sesión para cargar el contexto.

---

## Auditoría de seguridad

El comando `/8-auditar` (skill `audit-code` en `.claude/skills/`) ejecuta una auditoría de seguridad profesional antes de `/9-documentar`. Es un paso **obligatorio** del flujo de fase cuando el código toca:

- Autenticación / autorización
- Criptografía o almacenamiento de credenciales
- Entrada externa (correo, web, archivos subidos por usuario)
- `subprocess` / ejecución de comandos
- Deserialización (pickle, yaml.load, json.loads sobre input no validado)
- Dependencias nuevas o actualizadas

La auditoría produce un informe en `docs/security/audit-YYYY-MM-DD-<modo>.md` con severidad, CWE, OWASP, file:line y fix propuesto. Cada hallazgo bloqueante (Critical/High) se resuelve con un `fix-N` antes de cerrar la fase. El catálogo consolidado vive en `docs/security/README.md`.

Ver `CLAUDE_GLOBAL.md` → sección "Auditoría de seguridad" para las reglas detalladas.
