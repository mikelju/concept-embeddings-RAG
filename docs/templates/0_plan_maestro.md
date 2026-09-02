# Plan Maestro: [Nombre del Producto]

## ¿Qué es este producto?

[2-3 frases: problema que resuelve, para quién, cómo lo resuelve]

## Convención de documentación

Toda modificación, mejora o corrección sigue este protocolo antes de tocar código:

```
docs/plans/
├── 0_plan_maestro.md          # Este archivo — visión global
├── fase_1/
│   ├── 1.spec.md              # Especificación funcional (QUÉ + POR QUÉ) — /4-especificar
│   ├── 1.0_nombre_fase.md     # Plan de implementación (CÓMO) — /5-planear
│   ├── 1.tasks.md             # (Opcional) Tareas atómicas — solo fases complejas
│   └── 1.Y_nombre.md          # Desviación/problema correlativo (Y = 1, 2, 3…)
├── fase_2/
│   └── ...
└── fixes/
    └── fix-N_nombre.md        # Bug fix puntual (correlativo global)
```

- **`X.spec.md`** → Especificación funcional. Se crea con `/4-especificar` ANTES de planificar.
- **`X.0`** → Plan de implementación. Se crea con `/5-planear` DESPUÉS de la spec.
- **`X.tasks.md`** → Tareas atómicas. Opcional, solo si la fase tiene >10 pasos.
- **`X.Y`** → Desviación, problema inesperado o ajuste fuera del plan (Y correlativo).

**Flujo de trabajo por fase:** `/4-especificar` → `/5-planear` → `/6-implementar` → `/7-verificar`

---

## Estado de fases

| Fase | Nombre | Spec | Estado |
|------|--------|------|--------|
| 1 | [Nombre] | Pendiente | Pendiente |
| 2 | [Nombre] | Pendiente | Pendiente |
| 3 | [Nombre] | Pendiente | Pendiente |

---

## Fase 1: [Nombre]
- [ ] [Hito 1]
- [ ] [Hito 2]
- [ ] [Hito 3]

## Fase 2: [Nombre]
- [ ] [Hito 1]
- [ ] [Hito 2]

## Fase 3: [Nombre]
- [ ] [Hito 1]
- [ ] [Hito 2]

---

## Correctivos
[Referencias a fix-N de funcionalidades sin plan de fase propio]
