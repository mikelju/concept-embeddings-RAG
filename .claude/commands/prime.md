# Comando /prime — Carga de contexto inicial

Ejecuta este protocolo completo al inicio de cada sesión, antes de cualquier tarea.
El objetivo es construir un mapa mental preciso del proyecto en su estado actual.

---

## Fase 1: Reglas y arquitectura del proyecto

Lee `CLAUDE.md` en la raíz del proyecto.

Extrae y memoriza:
- Nombre del proyecto y qué hace
- Stack tecnológico y versiones
- Comandos de desarrollo y cómo arrancar el proyecto
- Decisiones de arquitectura y patrones principales
- Permisos del agente (SIEMPRE / PREGUNTAR / NUNCA)
- Anti-patrones del proyecto
- Gotchas conocidos
- Convenciones de código

---

## Fase 2: Estado del plan maestro

Lee `docs/plans/0_plan_maestro.md`.

Identifica:
- Qué fases existen y cuál es su estado (pendiente / en curso / completada)
- Cuál es la fase activa actual (la primera no completada)
- Si hay subcarpetas de fase creadas en `docs/plans/`

---

## Fase 3: Spec y plan de la fase activa

Para la fase activa (la primera no completada):

1. **Comprobar si existe `X.spec.md`:**
   - Si existe → leerla. Anotar criterios de aceptación pendientes vs completados.
   - Si NO existe → marcar como "⚠️ Sin spec funcional".

2. **Leer `X.0_*.md`** (plan principal de la fase):
   - Determinar qué tareas están completadas (`[x]`) y cuáles pendientes (`[ ]`).

3. **Si existe `X.tasks.md`** → leerlo para conocer el desglose atómico de tareas.

Solamente si el usuario quiere info específica de una fase, leer adicionalmente:

4. Archivos `X.Y_*.md` (desviaciones o ajustes).

---

## Fase 4: Historial de cambios recientes

Ejecuta los siguientes comandos para entender qué ha cambiado recientemente:

```bash
git log --oneline -10
git status
```

Identifica:
- Los últimos cambios realizados en el proyecto
- Si hay trabajo en progreso sin commitear

---

## Fase 5: Archivos clave del código

En `CLAUDE.md` (sección "Estructura de archivos") están listados los archivos principales del proyecto. Lee cada uno de ellos.

Si `CLAUDE.md` no lista archivos explícitamente, infiere los puntos de entrada habituales según el stack (ej. el servidor principal, el componente raíz del frontend, el archivo de configuración de base de datos, etc.).

Nota mentalmente:
- Funciones principales y su propósito
- Cualquier TODO, comentario de deuda técnica o código provisional

---

## Entrega: Informe de contexto

Al finalizar las 5 fases, presenta al usuario un informe estructurado:

```
## Estado del proyecto: [Nombre del proyecto]

### Fase activa
[Nombre y número de la fase en curso]

### Spec funcional
[✅ Disponible (X criterios pendientes de Y total) | ⚠️ No existe — ejecutar /4-especificar]

### Tareas pendientes en la fase activa
- [ ] Tarea 1
- [ ] Tarea 2

### Cambios recientes (git)
[Últimos 3-5 commits relevantes]

### Trabajo sin commitear
[Archivos modificados, si los hay]

### Próximo paso sugerido
[Una sola acción concreta y específica]
```

Solo después de entregar este informe, preguntar al usuario qué quiere hacer.
