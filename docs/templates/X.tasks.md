# Tareas Fase X: [Nombre de la Fase]

> Spec: `X.spec.md` | Plan: `X.0_nombre.md`
> Este archivo es OPCIONAL — solo para fases con >10 pasos de implementación.

<!--
Cada tarea debe ser:
- Atómica: corresponde a ~1 commit
- Pequeña: afecta a máximo 3-5 archivos
- Verificable: tiene criterios claros de "completado"
- Ordenada: respeta las dependencias entre tareas

Marcadores:
- [Story: HU-X] → vincula con la historia de usuario de la spec
- [depende de TX] → no puede empezar hasta que TX esté completada
- [P] → puede ejecutarse en paralelo con otras tareas
-->

---

## T1: [Nombre descriptivo] [Story: HU-1]

- **Archivos**: `ruta/archivo1.ts`, `ruta/archivo2.ts`
- **Acción**: [qué hacer concretamente en esta tarea]
- **Verificación**:
  - [ ] [cómo comprobar que funciona]
  - [ ] [test específico que debe pasar]

## T2: [Nombre descriptivo] [Story: HU-1] [depende de T1]

- **Archivos**: `ruta/archivo.ts`
- **Acción**: [qué hacer concretamente]
- **Verificación**:
  - [ ] [comprobación]

## T3: [Nombre descriptivo] [Story: HU-2] [P]

- **Archivos**: `ruta/archivo.ts`, `tests/archivo.test.ts`
- **Acción**: [qué hacer concretamente]
- **Verificación**:
  - [ ] [comprobación]
  - [ ] [comprobación]

## T4: [Nombre descriptivo] [Story: HU-2] [depende de T2, T3]

- **Archivos**: `ruta/archivo.ts`
- **Acción**: [qué hacer concretamente]
- **Verificación**:
  - [ ] [comprobación]

---

## Resumen de estado

| Tarea | Story | Estado |
|-------|-------|--------|
| T1 | HU-1 | Pendiente |
| T2 | HU-1 | Pendiente |
| T3 | HU-2 | Pendiente |
| T4 | HU-2 | Pendiente |
