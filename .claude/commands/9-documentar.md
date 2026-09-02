# Comando /9-documentar

Genera la documentación de usuario de la aplicación. Es una guía práctica para la persona que va a USAR la aplicación, no para quien la desarrolla.

Se ejecuta al terminar una fase o al final del proyecto, cuando la funcionalidad está verificada.

---

## Principios de la documentación

- **Audiencia**: usuario final, no técnico. No sabe qué es una API, un endpoint ni una base de datos.
- **Brevedad**: lo mínimo necesario para usar la aplicación con garantías. Si sobra una frase, se elimina.
- **Formato**: bullet points siempre que sea posible. Párrafos solo cuando sea imprescindible.
- **Emojis**: usar para marcar tipos de información y facilitar el escaneo visual.
- **Consultable**: estructura clara con secciones y subsecciones para encontrar rápido lo que se busca.
- **Ejemplos**: incluir ejemplos concretos cuando la explicación textual no sea suficiente.

---

## Protocolo

### Paso 1: Cargar contexto

Lee en este orden:

1. **`CLAUDE.md`** → nombre del producto, qué hace, para quién.
2. **Todas las specs completadas** (`docs/plans/fase_X/X.spec.md`) → historias de usuario y criterios de aceptación.
3. **El código** → pantallas, rutas, menús, formularios, mensajes de error reales.

### Paso 2: Determinar alcance

- Si el usuario dice "documenta la fase 2" → solo la funcionalidad de esa fase.
- Si dice "documenta todo" → toda la aplicación.
- Si no indica → documentar todas las fases completadas (con spec verificada).

### Paso 3: Extraer funcionalidades desde las specs

Para cada spec completada:

1. Leer las **historias de usuario** (HU-1, HU-2...).
2. Convertir cada historia en una sección de la guía desde la perspectiva del usuario.
3. Los **criterios de aceptación** se convierten en comportamientos que el usuario debe conocer (límites, validaciones, reglas).
4. Las **restricciones** relevantes para el usuario se incluyen como notas.
5. Los **anti-objetivos** se ignoran (son internos, no le importan al usuario).

### Paso 4: Revisar la aplicación real

No documentar desde la spec solamente — verificar contra el código real:

- ¿Qué pantallas/vistas existen?
- ¿Qué formularios hay y qué campos tienen?
- ¿Qué mensajes de error puede ver el usuario?
- ¿Hay flujos con varios pasos (wizards, procesos)?
- ¿Hay atajos o funcionalidades no obvias?

### Paso 5: Generar la documentación

Escribir el archivo `docs/GUIA_USUARIO.md` (o el nombre que indique el usuario) siguiendo la estructura de abajo.

### Paso 6: Revisión

Presentar al usuario para revisión:
> "Guía de usuario generada. ¿Hay algo incorrecto, que falte o que sobre?"

Incorporar correcciones.

---

## Estructura de la guía de usuario

```markdown
# [Nombre de la Aplicación] — Guía de Usuario

> [1 frase: qué hace esta aplicación y para quién]

---

## 📋 Índice

- [Primeros pasos](#primeros-pasos)
- [Sección por funcionalidad...]
- [Preguntas frecuentes](#preguntas-frecuentes)

---

## 🚀 Primeros pasos

[Los 3-5 pasos mínimos para empezar a usar la aplicación]

1. ...
2. ...
3. ...

---

## [Emoji] [Nombre de la funcionalidad]

[1 frase: qué puedes hacer aquí]

### Cómo [acción principal]

1. [paso]
2. [paso]
3. [paso]

### Lo que debes saber

- [regla o límite importante — derivado de criterios de aceptación]
- [otra regla]
- [otra regla]

### Ejemplo

> [ejemplo concreto de uso, si ayuda a entender]

---

## ❓ Preguntas frecuentes

**¿[pregunta que el usuario se haría]?**
[respuesta breve]

**¿[otra pregunta]?**
[respuesta breve]

---

## ⚠️ Problemas conocidos

- [limitación que el usuario debe conocer]
- [otra limitación]
```

---

## Reglas de redacción

### Sí hacer:
- Usar **tú** (tutear al usuario), no "usted"
- Verbos en imperativo: "Pulsa", "Escribe", "Selecciona"
- Ejemplos concretos: "Escribe tu email, por ejemplo: maria@ejemplo.com"
- Emojis como marcadores de sección: 🚀 📋 ✏️ 🔍 📊 ⚠️ ❓ 💡 🔒
- Bullet points para listas de reglas, pasos o condiciones
- Negritas para acciones clave: "Pulsa **Guardar**"
- Capturas de pantalla si el usuario las proporciona

### No hacer:
- No usar jerga técnica: ni "endpoint", ni "request", ni "autenticación OAuth"
- No explicar cómo funciona internamente — solo cómo se usa
- No incluir información para desarrolladores
- No escribir párrafos largos — si tiene más de 3 líneas, convertir a lista
- No documentar funcionalidades que no estén implementadas y verificadas
