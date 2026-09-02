# Spec Fase X: [Nombre de la Fase]

> Especificación funcional — QUÉ construir y POR QUÉ.
> El CÓMO se define en el plan de fase (X.0_nombre.md).
> Los criterios de aceptación se verifican con `/7-verificar` al terminar la fase.

---

## Problema y objetivo

**Problema**: [1-2 frases: qué problema del usuario resuelve esta fase]

**Objetivo**: [1 frase: cómo luce el éxito al terminar esta fase]

---

## Historias de usuario

### HU-1: [Nombre descriptivo]
**Como** [tipo de usuario],
**quiero** [acción],
**para** [beneficio].

**Criterios de aceptación:**
- [ ] [condición medible — pasa o no pasa]
- [ ] [condición medible]
- [ ] [condición medible]

### HU-2: [Nombre descriptivo]
**Como** [tipo de usuario],
**quiero** [acción],
**para** [beneficio].

**Criterios de aceptación:**
- [ ] [condición medible]
- [ ] [condición medible]

---

## Contratos de datos

<!-- Omitir esta sección en fases pequeñas (<1 día de trabajo) -->

### [Modelo / Tabla / Esquema principal]

| Campo | Tipo | Restricciones |
|-------|------|---------------|
| id | UUID/INT | PK, auto |
| ... | ... | ... |

**Invariantes:**
- [Regla de negocio que siempre debe cumplirse sobre estos datos]

### Endpoints

<!-- Omitir si la fase no expone API -->

**[MÉTODO] [ruta]**
- Descripción: [qué hace]
- Request: `{ campo: tipo }`
- Response 200: `{ campo: tipo }`
- Response error: `{ error: "mensaje", code: "CODIGO" }`

---

## Restricciones

- [Restricción técnica: BD existente, API externa, rendimiento...]
- [Restricción de negocio: regulación, formato de datos, limitaciones...]

---

## Anti-objetivos (lo que esta fase NO hace)

- NO [funcionalidad que podría confundirse con el alcance pero queda fuera]
- NO [otra cosa que queda explícitamente fuera]

---

## Contexto del código existente

<!-- Omitir si es un proyecto nuevo sin código previo -->

- `ruta/archivo` — [qué existe y qué relación tiene con esta spec]
- `ruta/otro` — [dependencia relevante que el agente debe conocer]

<!--
GUÍA DE NIVEL DE DETALLE:

Fase pequeña (<1 día): Problema + objetivo, 1-2 HU con criterios, anti-objetivos.
Fase media (1-5 días): Todas las secciones. Contratos con campos principales.
Fase grande (>5 días): Todas las secciones con máximo detalle. Esquemas JSON explícitos.

Regla: si la fase toca >5 archivos, incluir contratos de datos completos.
-->
