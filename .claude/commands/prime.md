# Comando /prime — Carga de contexto inicial

Orientación barata al empezar una sesión. `CLAUDE.md` ya está en contexto: no lo releas.
El objetivo es saber dónde está el proyecto, no leerlo entero.

---

## 1. Estado del plan

Lee **solo** la sección "Phase status" de `docs/plans/0_master_plan.md` (la tabla y los párrafos
que la siguen), no el documento completo. Identifica la fase o desviación activa.

## 2. Documento activo

De la fase activa en `docs/plans/phase_X/`, lee la cabecera (estado y objetivo) de su spec y los
pasos pendientes (`[ ]`) de su plan `X.0_*.md`, si existe. No leas los `X.results.md` de fases
cerradas ni los planes de implementación ya ejecutados salvo que la tarea lo pida.

## 3. Git

```bash
git branch --show-current
git log --oneline -8
git status --short
gh pr list --state open --limit 5
```

## 4. Código

No leas ficheros de código por adelantado. Léelos cuando la tarea los necesite.

---

## Entrega

```
## Estado: concept-embeddings-RAG

Fase activa: [nombre] — [spec: aprobada | borrador | no existe]
Pasos pendientes: [lista corta o "ninguno"]
Rama: [rama] — [trabajo sin commitear, si lo hay]
PR abiertas: [número y título, o "ninguna"]
Próximo paso sugerido: [una acción concreta]
```

Después, pregunta qué se quiere hacer.
