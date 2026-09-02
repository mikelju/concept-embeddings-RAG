# Fix N: [Titulo del Bug]

<!-- N es un número correlativo GLOBAL (fix-1, fix-2, fix-3...), compartido entre todas las fases. -->

## Fase / funcionalidad afectada
Qué parte del producto tiene el bug y a qué fase del plan pertenece.

<!-- Si este fix cierra un hallazgo de seguridad, añadir aquí la referencia:
Referencia al hallazgo: [docs/security/audit-YYYY-MM-DD-fase-X.md §3.N SEC-NNN](../security/audit-YYYY-MM-DD-fase-X.md)
-->

## Síntoma
Qué comportamiento incorrecto observa el usuario (o, si es un hallazgo de seguridad, el vector explotable).

## Causa raíz
Por qué ocurre exactamente (con referencias a archivo:linea si aplica).

## Solución adoptada
Qué se cambió exactamente.

## UX
<!-- Obligatorio si el fix toca algo visible al usuario. Escribir "Sin cambio"
si el arreglo es interno y el comportamiento legítimo no se altera. Este campo
es especialmente importante para fixes de seguridad, donde el objetivo es
cerrar el vector sin introducir fricción al usuario final. -->
[Sin cambio / qué nota el usuario]

## Archivos modificados
- `ruta/archivo:linea` — descripción del cambio

## Referencia cruzada
<!-- Obligatorio: este fix DEBE quedar referenciado en la sección "Correctivos"
del plan de fase afectado (X.0_nombre.md). Si la fase no tiene plan, referenciarlo
en la sección "Correctivos" del plan maestro.

Si el fix cierra uno o varios hallazgos de seguridad, listar aquí los IDs
cerrados para mantener trazabilidad en docs/security/README.md:
- Cierra: SEC-NNN, SEC-MMM, DEF-NNN (solo si aplica).
-->
- Registrado en: `docs/plans/fase_X/X.0_nombre.md` → sección Correctivos
- Cierra: [lista opcional de SEC/DEF/OBS de `docs/security/`]
