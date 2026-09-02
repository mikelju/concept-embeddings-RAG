---
name: gdoc
description: "Convert a Markdown file to a Google Doc preserving rendered formatting (headings, tables, lists, code blocks, bold/italic). Use when the user wants to upload or convert a .md file to Google Docs."
---

# /gdoc — Convertir Markdown a Google Doc

Convierte un archivo `.md` a Google Doc, preservando el formato renderizado (encabezados, tablas, listas, negrita, cursiva, bloques de codigo, etc.). El documento se comparte automaticamente con el usuario.

## Uso

```
/gdoc <ruta.md>
/gdoc <ruta.md> --title "Titulo personalizado"
/gdoc <ruta.md> --folder-id <id_carpeta_drive>
```

## Protocolo

### 1. Identificar el archivo

- Si el usuario proporciona una ruta, verificar que existe
- Si no se proporciona ruta, preguntar que archivo quiere convertir
- Si es un nombre sin ruta, buscar en `output/entregables/`, `output/`, y la raiz del proyecto

### 2. Ejecutar la conversion

El script esta en `references/md_to_gdoc.py` dentro de este skill. Ejecutar desde la raiz del proyecto:

```bash
python .claude/skills/gdoc/references/md_to_gdoc.py <ruta-al-archivo.md> [opciones]
```

**Opciones disponibles:**

| Opcion | Descripcion |
|--------|-------------|
| `--title "Titulo"` | Titulo del Google Doc (default: nombre del archivo) |
| `--folder-id <id>` | Carpeta destino en Google Drive |
| `--share <email>` | Email para compartir (default: ${GDOC_SHARE_EMAIL}) |
| `--no-share` | No compartir automaticamente |
| `--keep-docx` | Mantener el .docx intermedio |
| `--credentials <ruta>` | Ruta a credentials.json (si no esta en la raiz) |

### 3. Confirmar resultado

- Mostrar al usuario la URL del Google Doc creado
- Si hubo error, diagnosticar:

| Error | Solucion |
|-------|----------|
| `credentials.json` no encontrado | Colocar el archivo Service Account en la raiz del proyecto |
| Google Drive API no habilitada | Habilitar en Google Cloud Console |
| Error de permisos al compartir | Mostrar la URL directa del documento |
| Dependencias faltantes | `pip install python-docx markdown-it-py google-api-python-client google-auth` |

## Setup (una vez por proyecto)

1. Colocar `credentials.json` (Service Account de Google) en la raiz del proyecto
2. Habilitar Google Drive API en el proyecto de Google Cloud
3. Instalar dependencias: `pip install python-docx markdown-it-py google-api-python-client google-auth`

## Como funciona internamente

1. **MD → DOCX**: Usa `md_to_docx.py` para convertir el Markdown a Word con formato completo
2. **DOCX → Google Doc**: Sube el .docx a Google Drive con conversion automatica a formato nativo
3. **Compartir**: Da permisos de editor al email configurado
4. **Limpiar**: Elimina el .docx temporal
