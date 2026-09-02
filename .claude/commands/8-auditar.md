# Comando /8-auditar

Auditoría de seguridad profesional sobre el código. Se apoya en la skill `audit-code` (`.claude/skills/audit-code/`) y produce un informe reproducible con hallazgos, severidad, CWE, OWASP y fix propuesto para cada issue.

Es el paso **8** del flujo de fase: se ejecuta **después** de `/7-verificar` (cuando los tests ya pasan y el código está alineado con la spec) y **antes** de `/9-documentar` (para que la documentación refleje el estado final y seguro).

---

## Cuándo usar este comando

- **Obligatorio al cerrar una fase** que toque código de seguridad (autenticación, criptografía, entrada de usuario, dependencias externas, manejo de ficheros, subprocess, deserialización).
- **Obligatorio antes de cualquier release** (modo `completo`).
- **Recomendado tras un fix** en módulos sensibles.
- **Ad-hoc** cuando el usuario pida una revisión de seguridad.

No sustituye a `/7-verificar` (que valida funcionalidad contra la spec). Este comando valida **seguridad**.

---

## Modos

El modo por defecto es **fase activa**: se audita solo lo modificado en la fase actual (scope acotado, feedback rápido, integrado en el flujo SDD).

| Modo | Qué audita | Cuándo usarlo |
|------|-----------|---------------|
| `/8-auditar` (sin argumentos) | Archivos de la **fase activa** (detectada en el plan maestro) | Al cerrar cada fase |
| `/8-auditar fase X` | Archivos de la fase X concreta | Re-auditoría puntual de una fase anterior |
| `/8-auditar completo` | Todo `src/` — release gate | Antes de empaquetar, desplegar, publicar |
| `/8-auditar rapido` | Solo escaneo automatizado (sin revisión manual) | Chequeo ligero o CI |
| `/8-auditar deps` | Solo dependencias (pip-audit, safety, npm audit) | Tras añadir/actualizar dependencias |
| `/8-auditar secretos` | Solo búsqueda de credenciales hardcoded | Tras una filtración o como pasada rutinaria |
| `/8-auditar <ruta>` | Ruta libre (archivo, directorio, glob) | Scope custom |

---

## Protocolo

### Paso 1: Cargar skill y contexto

1. Invocar la skill `audit-code` (o leer `.claude/skills/audit-code/SKILL.md` completo).
2. Leer `CLAUDE.md` del proyecto: stack, comandos, gotchas.
3. Leer `docs/plans/0_plan_maestro.md`: identificar la **fase activa** (primera no completada).
4. Si el modo es `fase X` (o por defecto) → leer `docs/plans/fase_X/X.spec.md` y `X.0_*.md` para conocer los archivos tocados en esa fase.

### Paso 2: Escape hatch — ¿esta fase necesita auditoría?

Antes de nada, comprobar si la fase activa es auditable. Si se cumple **cualquiera** de estos, preguntar al usuario si quiere saltarla:

- La fase no toca ningún archivo bajo `src/` (solo docs, configuración cosmética, assets).
- La fase es un refactor puro sin cambio de comportamiento (ojo: un refactor que toca auth, crypto o parsing de input **NO** es exento — auditar igual).
- La fase es una migración de ficheros/estructura sin lógica nueva.

Formular la pregunta así:

```
La fase {{X}} — "{{nombre}}" parece no contener código de seguridad relevante
(motivo: {{docs only / refactor sin cambio funcional / assets…}}).

¿Saltamos la auditoría para esta fase?
- Si respondes "sí" → registro nota "Fase X exenta — motivo: ..." en el informe
  y continúo con /9-documentar.
- Si respondes "no" → procedo con la auditoría completa igualmente.
```

Si el usuario confirma la exención, **crear un mini-informe** en `docs/security/audit-YYYY-MM-DD-fase-X.md` con el motivo documentado (dejar trazabilidad — una fase exenta no es una fase no auditada sin más, es una decisión consciente registrada).

Si la fase SÍ toca código sensible, nunca ofrecer el escape hatch.

### Paso 3: Definir y comunicar alcance

Presentar al usuario:

```
## Auditoría — Alcance

- **Scope:** {{archivos concretos de la fase X, o "src/ completo" si modo `completo`}}
- **Modo:** {{fase / completo / rapido / deps / secretos / ruta custom}}
- **Threat model asumido:** {{inferido del CLAUDE.md del proyecto}}
- **Stack detectado:** {{Python 3.11, …}}
- **Herramientas disponibles localmente:** bandit [sí/no], pip-audit [sí/no], semgrep [sí/no], gitleaks [sí/no], ruff [sí/no]
```

Si el usuario confirma o guarda silencio → continuar. Si corrige scope/threat model → ajustar.

### Paso 4: Ejecutar las 5 fases de la skill

Seguir **Phase 1 → Phase 5** de `SKILL.md`:

1. **Phase 1 — Scope & Context**: ya hecho en Paso 3.
2. **Phase 2 — Automated Scanning**: probar qué herramientas están instaladas (`references/tools-integration.md`) y ejecutar las que estén. **NO instalar nada sin autorización.**
3. **Phase 3 — Manual Review**: leer `references/python-security.md` completo. Si hay código React/JS/TS, también `references/react-security.md`. Aplicar las 10 preguntas por archivo.
4. **Phase 4 — Exploit Reasoning**: para cada hallazgo, razonar explotabilidad y redactar PoC conceptual (nunca ejecutar).
5. **Phase 5 — Report**: generar informe siguiendo `references/report-template.md`.

En modos reducidos (`rapido`, `deps`, `secretos`), ejecutar solo la fase correspondiente:
- `rapido` → Phases 1, 2 y 5 (saltar Manual Review).
- `deps` → solo escaneo de dependencias + informe parcial.
- `secretos` → solo scan de secretos + informe parcial.

### Paso 5: Coverage Self-Check

Antes de presentar el informe final, ejecutar el checklist de cobertura de `SKILL.md`. Si algún ítem no se ha cubierto → indicarlo explícitamente en la sección §5 Coverage del informe.

### Paso 6: Publicar informe

Ruta según modo:
- Modo `fase X` / por defecto → `docs/security/audit-YYYY-MM-DD-fase-X.md`
- Modo `completo` → `docs/security/audit-YYYY-MM-DD-completo.md`
- Modo `rapido`, `deps`, `secretos` → `docs/security/audit-YYYY-MM-DD-<modo>.md`

- Si `docs/security/` no existe → crear la carpeta.
- Si el informe ya existe para la misma combinación fecha+modo → sufijo `-2`, `-3`, etc.
- Si el usuario prefiere inline → mostrarlo en chat sin escribir archivo (pero sigue recomendado persistir).

Comunicar al usuario:
```
Auditoría completa. Informe en: docs/security/audit-YYYY-MM-DD-<modo>.md

Resumen:
- Critical: N (fix inmediato)
- High: N (antes de release)
- Medium: N
- Low: N
- Info: N

Top 3 a corregir primero:
1. [SEC-001] — <titulo>
2. [SEC-002] — <titulo>
3. [SEC-003] — <titulo>
```

### Paso 7: Oferta de remediación y cierre de fase

**No modificar código durante la auditoría.** Al terminar, preguntar:

> "¿Quieres que empiece a aplicar los fixes? Puedo abordarlos en orden de severidad (Critical → High → Medium). Para cada fix no trivial crearé un `fix-N` en `docs/plans/fixes/` siguiendo el protocolo del proyecto."

Si el usuario dice sí → cambiar de modo "auditor" a "remediador":
- Fixes triviales (una línea, un import) → aplicar directamente.
- Fixes no triviales → seguir flujo SDD: crear `docs/plans/fixes/fix-N_nombre.md` con plan mínimo, luego implementar.
- Rotar credenciales comprometidas **antes** de tocar nada más (esto lo hace el usuario; tú esperas confirmación).

**Cierre de fase:**
- Si no hay hallazgos **Critical ni High** → la fase puede cerrarse. Pasar a `/9-documentar`.
- Si hay Critical o High → la fase **no se cierra** hasta resolverlos o registrar decisión explícita de aplazar con justificación en el plan maestro.

---

## Reglas no negociables

1. **Nunca instalar herramientas** (`pip install`, `npm install`) sin autorización explícita.
2. **Nunca ejecutar payloads** contra el propio código ni contra sistemas externos. El PoC es descriptivo.
3. **Nunca volcar secretos completos** en el informe — redactar siempre (`sk-abcd•••`).
4. **Nunca decir "el código es seguro"** — usar "sin hallazgos críticos bajo el alcance revisado".
5. **Falsos positivos > falsos negativos**. Ante la duda, reportar como HIGH; el usuario baja si procede.
6. **Confirmar cada hallazgo automatizado leyendo el código** antes de incluirlo en el informe.
7. **No modificar código durante la auditoría** — solo después del informe y con luz verde explícita.
8. **Escape hatch** solo aplica a fases sin código de seguridad relevante. Nunca saltar una fase que toca auth, crypto, input externo, subprocess o deserialización.

---

## Salida esperada

- Archivo: `docs/security/audit-YYYY-MM-DD-<modo>.md` siguiendo la plantilla de `.claude/skills/audit-code/references/report-template.md`.
- Resumen en chat con contadores por severidad y top-3 prioridades.
- Oferta de remediación siguiendo el protocolo de fixes del proyecto.
- Decisión de cierre de fase (puede cerrarse / no puede cerrarse + motivo).
