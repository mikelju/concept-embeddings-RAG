# Comando /8-auditar

Revisión de seguridad del código de la fase: encuentra los problemas, los presenta ordenados por
gravedad y los arregla contigo en la misma sesión. No produce un informe que haya que procesar
después — el entregable es código corregido.

Es el paso **8** del flujo de fase: se ejecuta **después** de `/7-verificar` (los tests ya pasan y
el código está alineado con la spec) y **antes** de `/9-documentar` (para que la documentación
refleje el estado final y seguro).

---

## Cuándo usar este comando

- **Obligatorio al cerrar una fase** que toque autenticación, criptografía, entrada externa,
  dependencias, manejo de ficheros, subprocess o deserialización.
- **Obligatorio antes de un release** (modo `completo`).
- **Recomendado tras un fix** en módulos sensibles.
- **Ad-hoc** cuando se pida una revisión de seguridad.

No sustituye a `/7-verificar`, que valida funcionalidad contra la spec. Este comando valida
**seguridad**.

---

## Modos

Por defecto audita la **fase activa**: scope acotado, feedback rápido, integrado en el flujo SDD.

| Modo | Qué audita |
|---|---|
| `/8-auditar` | Archivos de la fase activa (la primera no completada en el plan maestro) |
| `/8-auditar fase X` | Una fase concreta — re-auditoría puntual |
| `/8-auditar completo` | Todo `src/` — release gate |
| `/8-auditar deps` | Solo dependencias: el escáner de CVEs del stack |
| `/8-auditar secretos` | Solo credenciales hardcoded |
| `/8-auditar <ruta>` | Scope libre: archivo, directorio o glob |

---

## Protocolo

### Paso 1: Alcance

1. `CLAUDE.md` ya está en contexto (stack, comandos, gotchas); no lo releas.
2. Leer `docs/plans/0_master_plan.md` para identificar la fase activa.
3. Leer la spec y el plan de esa fase para saber qué archivos toca.
4. Comprobar qué herramientas hay instaladas (`command -v <tool>`). **Nunca instalar nada.**

Comunicar antes de seguir:

```
## Auditoría — Fase X

- Alcance: <archivos concretos>
- Modo: <fase / completo / deps / secretos / ruta>
- Superficie sensible: <input externo, deserialización, subprocess, crypto...>
- Herramientas disponibles: <las que respondan; nombrar también las que faltan>
```

**Escape hatch.** Si la fase no toca `src/` (solo docs, assets o configuración cosmética), o es un
refactor puro sin cambio de comportamiento, o una migración de ficheros sin lógica nueva →
preguntar si se salta. Si el usuario acepta, anotar la exención con su motivo en
`docs/security/README.md` y pasar a `/9-documentar`: una fase exenta es una decisión registrada, no
una fase sin auditar sin más. Un refactor que toca auth, crypto o parsing de input **no** es
exento: ahí no se ofrece la salida.

### Paso 2: Escanear

Ejecutar los escáneres instalados que apliquen al stack: SAST, CVEs de dependencias, secretos.

- Cero hallazgos **no** cierra el paso. Los escáneres no ven fallos de lógica, de autorización ni
  de diseño; el paso 3 se ejecuta siempre.
- Cada hit se confirma **leyendo el código** antes de contarlo como hallazgo.
- Si falta una herramienta, decirlo en el resumen. No es un error: es cobertura que no se tiene.

### Paso 3: Leer el código

Para cada archivo del alcance:

1. **Entradas** — ¿de dónde viene el dato? ¿Validado, tipado y acotado en tamaño en el punto de
   uso? La confianza no es transitiva.
2. **Sinks** — `eval`, `exec`, `subprocess` con `shell=True`, `pickle`, `yaml.load`, SQL cruda,
   `os.system`, plantillas con dato de usuario, `dangerouslySetInnerHTML`.
3. **Autorización** — ¿hay comprobación en cada acción protegida? ¿Antes o después del efecto?
   ¿Se salta por orden, por carrera o manipulando parámetros (IDOR)?
4. **Salida** — ¿escapada para su contexto (HTML, URL, shell, SQL, log)? ¿Los errores filtran
   trazas, rutas o secretos?
5. **Crypto** — MD5/SHA1 para algo relevante, `random` en lugar de `secrets`, claves o IV fijos,
   verificación TLS desactivada.
6. **Secretos** — tokens, claves o cadenas de conexión en el código; secretos impresos o logueados.
7. **Dependencias** — paquetes sin mantenimiento, con CVE conocido, o añadidos en esta fase.
8. **Ficheros** — path traversal al leer, escribir o recibir subidas; zip-slip; symlinks; permisos.
9. **Concurrencia** — estado mutable compartido sin cerrojo, TOCTOU, doble envío.
10. **Integridad de artefactos** — lo que se lee de disco o de la red, ¿se **verifica** contra el
    hash registrado, o solo se registra?

### Paso 4: Presentar los hallazgos

En chat, no en un fichero. Ordenados de mayor a menor severidad:

```
### Hallazgos — Fase X

| ID | Sev | Dónde | Qué pasa | Fix |
|---|---|---|---|---|
| SEC-012 | High | `src/x.py:42` | <qué explota y qué gana el atacante> | <fix concreto> |

Sin hallazgos Critical/High en el alcance revisado.   <- si procede
No revisado: <lo que quedó fuera y por qué>
```

Cada hallazgo lleva severidad (rúbrica de `CLAUDE_GLOBAL.md`), un CWE cuando sea claro, y una
frase de explotabilidad: qué necesita el atacante y qué consigue. El PoC es descriptivo; nunca se
ejecuta.

### Paso 5: Arreglar

Preguntar antes de tocar: en bloque, o hallazgo a hallazgo de mayor a menor severidad.

- Cada fix lleva su test de regresión. Un fix sin test es un fix que vuelve.
- Fix no trivial → seguir la regla de fixes que ya tiene el proyecto (`/5-planear` →
  `/6-implementar`), no una propia de este comando.
- Si hay credenciales comprometidas, el usuario las rota **antes** de tocar nada más. Esperar su
  confirmación.
- Ejecutar los tests del proyecto al terminar. Nada se da por cerrado con tests en rojo.

### Paso 6: Anotar y cerrar

Una línea por hallazgo en `docs/security/README.md`, continuando la secuencia `SEC-NNN`: id,
severidad, título y estado. Ese catálogo es todo el papeleo de la auditoría.

Cierre de fase:

- Sin Critical ni High abiertos → la fase puede cerrarse. Pasar a `/9-documentar`.
- Con Critical o High → la fase **no se cierra** hasta resolverlos, o hasta registrar en el plan
  maestro la decisión explícita de aplazarlos con su justificación.

---

## Reglas no negociables

1. **Nunca instalar herramientas** sin autorización explícita.
2. **Nunca ejecutar payloads** contra el propio código ni contra sistemas externos.
3. **Nunca volcar un secreto completo** — redactar siempre (`sk-abcd•••`).
4. **Nunca decir "el código es seguro"** — decir "sin hallazgos altos en el alcance revisado".
5. **Falsos positivos antes que falsos negativos.** En la duda se reporta alto; el usuario baja.
6. **Confirmar leyendo el código** cada hallazgo que venga de una herramienta.
7. **No arreglar mientras se busca.** La lista se completa antes de tocar nada: arreglando sobre la
   marcha se pierde la visión ordenada por gravedad y se cambia el código bajo los pies del propio
   análisis.
8. **El escape hatch nunca se ofrece** en una fase que toca auth, crypto, input externo, subprocess
   o deserialización.

---

## Salida esperada

- Lista de hallazgos en chat, ordenada por severidad, con `file:line`, explotabilidad y fix.
- Código corregido con sus tests de regresión, y los tests del proyecto en verde.
- `docs/security/README.md` actualizado: una línea por hallazgo.
- Decisión de cierre de fase: puede cerrarse, o no puede y por qué.
