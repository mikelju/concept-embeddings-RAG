# Guía Definitiva de Spec-Driven Development (SDD)

## Cómo escribir especificaciones para que la IA construya exactamente lo que necesitas

---

## ¿Qué es esto del SDD y por qué debería importarme?

Imagina que contratas a un albañil extraordinariamente rápido. Puede levantar una casa en un día. El problema es que si le dices "hazme una casa bonita" sin darle planos, construirá algo... pero probablemente no lo que tú querías. Quizá ponga el baño en la cocina, o se olvide de las escaleras al segundo piso.

Eso es exactamente lo que pasa cuando usas agentes de IA para programar sin especificaciones claras. El agente es rápido y capaz, pero necesita planos precisos. **Spec-Driven Development (SDD) es la práctica de escribir esos planos antes de pedirle al agente que construya nada.**

En la práctica, SDD significa crear documentos de texto (normalmente en Markdown) donde describes con detalle qué quieres que haga tu software, qué reglas debe seguir, qué tecnologías usar y qué está prohibido. Estos documentos se convierten en la "fuente de verdad" del proyecto: si algo no está en la especificación, no existe.

### ¿Qué problema resuelve exactamente?

Hasta ahora, cuando programábamos con IA de forma conversacional (lo que se llama "vibe coding"), el proceso era algo así:

1. Le pides al agente: "Hazme un login"
2. Genera algo, pero no es lo que querías
3. Le dices: "No, con OAuth, no con contraseña"
4. Arregla eso, pero rompe otra cosa
5. Y así durante horas...

El historial de chat se pierde, el agente olvida instrucciones anteriores, y cada vez que retomas el proyecto partes de cero. Con SDD, todo el contexto está en documentos permanentes que el agente lee cada vez que trabaja. La diferencia es brutal:

| Aspecto | Vibe Coding (conversacional) | SDD (con especificaciones) |
|---------|-----|-----|
| **Fuente de verdad** | El historial del chat | Documentos de especificación |
| **Consistencia** | Baja — el agente olvida instrucciones | Alta — el contexto es permanente |
| **Escalabilidad** | Se rompe en proyectos con muchos archivos | Permite descomponer en tareas pequeñas |
| **Validación** | Manual — "probar hasta que funcione" | Automática — se comprueba contra la spec |
| **Tu rol** | Escritor de prompts | Arquitecto y validador |
| **Resultados** | Impredecibles | Alineados con el diseño |

### ¿Cuándo merece la pena usar SDD?

No todo necesita especificación formal. Hay un espectro:

- **Un script rápido o prototipo**: Vibe coding está bien. Pide y ajusta.
- **Una funcionalidad nueva en un proyecto existente**: Spec mínima (10 min). Ahorra horas de iteración.
- **Un microservicio o API completa**: Spec detallada (30–60 min). Imprescindible.
- **Un sistema distribuido con agentes en la nube**: Ecosistema completo de specs. Sin esto, el caos está garantizado.

La regla de oro que usa la comunidad: **"No haces vibe coding con cálculos financieros. No sobre-especificas un botón."**

---

## Los tres niveles de compromiso con SDD

No hace falta ir "all-in" desde el primer día. SDD tiene tres niveles de intensidad, y puedes elegir el que encaje con tu proyecto:

### Nivel 1: Spec-First (la especificación como punto de partida)

Escribes la especificación antes de programar, pero una vez que el código funciona, la spec puede quedar en segundo plano o incluso descartarse. Es el punto de entrada al paradigma. Ideal para funcionalidades aisladas o equipos que empiezan con SDD.

**Ejemplo**: Necesitas un endpoint de notificaciones. Escribes qué debe hacer, el agente lo construye, las pruebas pasan, y sigues adelante.

### Nivel 2: Spec-Anchored (la especificación como ancla permanente)

La especificación se mantiene viva durante toda la vida del proyecto. Si el código evoluciona, la spec se actualiza primero. Si el código diverge de la spec, algo falla automáticamente. Este es el nivel recomendado para proyectos de producción.

**Ejemplo**: Tu API de comunicación en la nube tiene 20 endpoints. La spec define cada uno. Si un desarrollador cambia un endpoint sin actualizar la spec, las pruebas de contrato fallan.

### Nivel 3: Spec-as-Source (la especificación ES el código fuente)

La posición más radical. El desarrollador solo edita la especificación, y el código se genera automáticamente a partir de ella. El código incluye advertencias como `// GENERATED FROM SPEC - DO NOT EDIT`. Si necesitas cambiar algo, lo cambias en la spec y regeneras.

**Ejemplo**: Defines las firmas de tus funciones en la spec, y el framework Tessl genera el código completo. El código es "desechable" — si la spec es sólida, regenerar es instantáneo.

**Para proyectos de agentes en la nube**, se recomienda apuntar al menos a **Spec-Anchored**. En estos entornos, la coordinación entre múltiples servicios hace imposible que un humano mantenga todo en la cabeza. La especificación viva garantiza que cualquier agente que toque el sistema comparta exactamente el mismo modelo mental.

---

## La estructura de archivos: tu ecosistema de especificaciones

El error más común al empezar con SDD es pensar que "la especificación" es un único documento gigante. En realidad necesitas un **ecosistema de documentos**, cada uno con un propósito claro. Esto no es solo por organización: los modelos de IA tienen una "ventana de contexto" limitada (la cantidad de texto que pueden procesar a la vez), y un documento masivo y desestructurado agota esa capacidad, haciendo que el agente ignore restricciones críticas.

### La estructura estándar

Así debería verse la carpeta de tu proyecto:

```
mi-proyecto/
├── .specify/
│   ├── memory/
│   │   └── constitution.md          # Las reglas sagradas del proyecto
│   ├── templates/
│   │   ├── spec-template.md         # Plantilla para nuevas specs
│   │   ├── plan-template.md         # Plantilla para planes técnicos
│   │   └── tasks-template.md        # Plantilla para listas de tareas
│   └── scripts/
├── specs/
│   ├── 001-autenticacion/
│   │   ├── spec.md                  # ¿QUÉ hace esta funcionalidad?
│   │   ├── plan.md                  # ¿CÓMO lo implementamos?
│   │   ├── tasks.md                 # ¿QUÉ PASOS concretos hay que dar?
│   │   ├── data-model.md            # Esquemas de base de datos
│   │   └── contracts/               # Contratos de API (OpenAPI, AsyncAPI)
│   ├── 002-notificaciones/
│   └── 003-facturacion/
├── CLAUDE.md                        # Instrucciones para Claude Code
├── AGENTS.md                        # Instrucciones universales para agentes
└── src/                             # Código fuente (generado desde las specs)
```

Las carpetas de funcionalidad usan **numeración secuencial** (`001-nombre`, `002-nombre`) y las ramas de Git pueden reflejar estos nombres. Para proyectos grandes, puedes separar specs por dominio: `SPEC_backend.md`, `SPEC_frontend.md`, `SPEC_database.md`.

La jerarquía fluye siempre hacia abajo: la constitución gobierna todas las specs → cada spec alimenta su plan → cada plan alimenta sus tareas → cada tarea referencia todo lo anterior.

---

## Documento 1: La Constitución (`constitution.md`)

### Qué es y para qué sirve

La constitución es el documento más importante y el más estable de todo tu proyecto. Define las reglas que **nunca cambian**, independientemente de qué funcionalidad estés construyendo. Piensa en ella como la "ley fundacional" de tu repositorio. El agente de IA la lee **antes de hacer cualquier cosa**.

La constitución no dice QUÉ hace tu software. Dice CÓMO debe construirse todo en tu proyecto.

### Qué debe incluir (con ejemplos)

#### 1. Comandos ejecutables exactos

No digas "ejecuta los tests". Di exactamente cómo:

```markdown
## Comandos del proyecto

- **Ejecutar tests**: `npm test -- --coverage`
- **Linting**: `eslint src/ --ext .ts,.tsx --fix`
- **Build**: `npm run build`
- **Arrancar en desarrollo**: `npm run dev`
- **Migrar base de datos**: `npx prisma migrate dev`
```

#### 2. Estructura del proyecto

El agente necesita saber dónde vive cada cosa para no crear archivos en lugares incorrectos:

```markdown
## Estructura de directorios

src/
├── routes/        # Definiciones de rutas Express (SIN lógica de negocio)
├── services/      # Lógica de negocio
├── data-clients/  # Acceso a bases de datos y APIs externas
├── middleware/     # Middleware compartido (auth, errores, logging)
├── types/         # Interfaces y tipos TypeScript
└── utils/         # Funciones auxiliares puras
```

#### 3. Estilo de código con ejemplo real

Un snippet vale más que mil palabras. En vez de escribir "usa funciones flecha, exportaciones nombradas y tipos explícitos", muestra un ejemplo:

```markdown
## Estilo de código — Ejemplo de referencia

Todo archivo nuevo debe seguir este patrón:

\```typescript
// ✅ ASÍ se escribe código en este proyecto
import { Request, Response } from 'express';
import { UserService } from '../services/user.service';

export const getUserById = async (req: Request, res: Response) => {
  const { id } = req.params;
  const user = await UserService.findById(id);
  
  if (!user) {
    return res.status(404).json({ error: 'User not found', code: 'USER_NOT_FOUND' });
  }
  
  return res.status(200).json({ data: user });
};
\```
```

#### 4. Patrones prohibidos (anti-patrones)

Esto es **crucial**. Los modelos de IA tienden a usar soluciones genéricas que pueden violar las convenciones de tu equipo. Decirle qué NO hacer es a menudo más efectivo que decirle qué hacer:

```markdown
## Lo que NUNCA debe hacerse

🚫 **NUNCA** usar bloques try/catch dentro de los routers.
   El manejo de errores está centralizado en el middleware `errorHandler.ts`.

🚫 **NUNCA** commitear claves secretas o tokens.
   Todas las credenciales van en variables de entorno (.env).

🚫 **NUNCA** añadir dependencias nuevas sin aprobación.
   Antes de hacer `npm install`, marcar como [REQUIERE APROBACIÓN].

🚫 **NUNCA** eliminar tests que estén fallando sin resolver el problema.

🚫 **NUNCA** escribir lógica de negocio directamente en los routers.
   Los routers solo llaman a servicios.
```

#### 5. El sistema de permisos: Siempre / Preguntar / Nunca

Este es uno de los patrones más potentes de SDD. Clasifica las acciones del agente en tres categorías:

```markdown
## Permisos del agente

✅ **SIEMPRE** (hacer sin preguntar):
- Ejecutar tests antes de dar por completada una tarea
- Seguir las convenciones de nombrado del proyecto
- Incluir tipos TypeScript en toda función nueva
- Añadir logs estructurados en endpoints nuevos

⚠️ **PREGUNTAR PRIMERO** (suspender y pedir confirmación):
- Modificar el esquema de base de datos
- Añadir nuevas dependencias a package.json
- Cambiar configuración de CI/CD
- Crear tablas o colecciones nuevas
- Modificar middleware compartido

🚫 **NUNCA** (prohibido sin excepción):
- Commitear secretos o credenciales
- Editar node_modules/ o archivos generados
- Borrar tests sin resolverlos
- Cambiar la versión de Node.js o TypeScript
- Conectar directamente a la base de datos desde los routers
```

#### 6. Cumplimiento normativo (si aplica)

Para proyectos que manejan datos personales o están en sectores regulados:

```markdown
## Cumplimiento y seguridad

- Todo dato personal (PII) debe almacenarse cifrado en reposo (AES-256)
- Los logs NUNCA deben contener emails, nombres o DNIs
- Cumplimiento GDPR: todo endpoint que devuelva datos de usuario
  debe soportar el derecho de borrado
- Las APIs externas se llaman siempre a través del data-client,
  nunca directamente desde servicios
```

---

## Documento 2: La Especificación Funcional (`spec.md`)

### Qué es y para qué sirve

Si la constitución define CÓMO se construye, la especificación funcional define QUÉ se construye y POR QUÉ. Es el contrato entre tu intención como desarrollador y la ejecución del agente. Este documento se centra en el resultado final, no en los detalles técnicos de cómo lograrlo.

Cada funcionalidad nueva (login, notificaciones, facturación...) tiene su propio `spec.md` en su carpeta correspondiente.

### Estructura recomendada

#### Cabecera con metadatos (frontmatter YAML)

Algunos frameworks (como Tessl) usan un bloque de metadatos al inicio del archivo. Aunque no uses esas herramientas, es buena práctica incluirlo porque limita el "radio de acción" del agente:

```yaml
---
name: Sistema de Notificaciones Push
description: Envío de notificaciones en tiempo real a usuarios conectados
status: draft
targets:
  - src/services/notification*
  - src/routes/notification*
  - tests/notification*
---
```

El campo `targets` es especialmente útil: le dice al agente exactamente qué archivos puede tocar, evitando que modifique áreas no relacionadas del proyecto.

#### Sección "Por qué" (el problema)

```markdown
## Problema

Los usuarios no reciben actualizaciones de estado de sus pedidos en tiempo real.
Actualmente tienen que refrescar la página manualmente, lo que genera
frustración y llamadas al soporte (un 23% de los tickets son "¿dónde está mi pedido?").

## Objetivo

Implementar un sistema de notificaciones push que informe al usuario
en tiempo real cuando su pedido cambie de estado.
```

#### Historias de usuario con criterios de aceptación

```markdown
## Historias de Usuario

### HU-1: Recibir notificación de cambio de estado
**Como** comprador registrado,
**quiero** recibir una notificación instantánea cuando mi pedido cambie de estado,
**para** saber en todo momento dónde está sin tener que comprobar manualmente.

**Criterios de aceptación:**
- [ ] El usuario recibe la notificación en menos de 5 segundos tras el cambio
- [ ] La notificación muestra: nombre del pedido, estado anterior y estado nuevo
- [ ] Si el usuario no está conectado, la notificación se almacena y se entrega al reconectar
- [ ] El usuario puede marcar notificaciones como leídas
- [ ] Tasa de entrega exitosa: ≥ 99,9% tras reintentos

### HU-2: Gestionar preferencias de notificación
**Como** usuario registrado,
**quiero** poder activar o desactivar tipos de notificación,
**para** recibir solo las que me interesan.

**Criterios de aceptación:**
- [ ] Existe un endpoint GET /api/notifications/preferences
- [ ] Existe un endpoint PUT /api/notifications/preferences
- [ ] Las preferencias se aplican inmediatamente (sin necesidad de reconectar)
```

Los criterios de aceptación deben ser **binarios**: se cumple o no se cumple. Nada de "debería ser razonablemente rápido" — di "en menos de 5 segundos".

#### Restricciones y "anti-objetivos"

```markdown
## Restricciones

- Las notificaciones se envían a través de WebSocket, no polling
- Máximo 3 reintentos antes de descartar una notificación fallida
- No se almacenan notificaciones de más de 30 días

## Anti-objetivos (lo que este sistema NO hace)

- NO gestiona el envío de emails (eso es otro servicio)
- NO implementa notificaciones SMS
- NO modifica el flujo de pedidos existente
```

Los anti-objetivos son clave para evitar que la IA "se emocione" y añada funciones que no le has pedido.

#### Estado actual del código

Si estás trabajando sobre un proyecto existente (brownfield), incluye un mapeo de qué existe ya:

```markdown
## Contexto del código existente

- `src/services/order.service.ts`: Ya tiene un método `updateStatus()`
  que debe emitir el evento de notificación (NO crear uno nuevo)
- `src/middleware/auth.ts`: Middleware de autenticación JWT ya funcional
- `src/data-clients/redis.client.ts`: Cliente Redis ya configurado,
  usar para almacenar notificaciones pendientes
- Base de datos: PostgreSQL con tabla `orders` existente
```

### Cómo escribir requisitos que la IA entienda sin ambigüedad: la notación EARS

Los modelos de IA son estadísticos y susceptibles a la ambigüedad del lenguaje humano. Si escribes requisitos como si fueran un ensayo literario, el resultado será errático. Para que la spec funcione como un conjunto de instrucciones precisas, existe un estándar llamado **EARS** (Easy Approach to Requirements Syntax) que usa plantillas fijas para estructurar cada regla.

La idea es simple: en vez de escribir prosa libre, usas patrones predefinidos:

| Tipo | Plantilla | Cuándo usarla | Ejemplo |
|------|-----------|---------------|---------|
| **Siempre** | `El <sistema> DEBE <respuesta>` | Comportamientos que siempre aplican | "El gateway de autenticación DEBE validar tokens JWT en cada petición" |
| **Evento** | `CUANDO <algo ocurra>, el <sistema> DEBE <respuesta>` | Funcionalidades reactivas | "CUANDO un pedido cambie de estado, el servicio de notificaciones DEBE enviar un mensaje WebSocket al usuario" |
| **Estado** | `MIENTRAS <condición>, el <sistema> DEBE <respuesta>` | Comportamiento durante un estado concreto | "MIENTRAS el broker de mensajería esté caído, el servicio DEBE almacenar mensajes en la cola local" |
| **Opcional** | `SI <funcionalidad está activa>, el <sistema> DEBE <respuesta>` | Funciones que pueden estar activadas o no | "SI la telemetría está habilitada, el agente DEBE exportar métricas cada 30 segundos" |
| **Error** | `SI <fallo>, ENTONCES el <sistema> DEBE <respuesta>` | Manejo de fallos y excepciones | "SI la conexión WebSocket se pierde, ENTONCES el cliente DEBE reconectar con backoff exponencial" |

Al forzar tus requisitos en estas plantillas, eliminas las vaguedades. El agente no necesita "adivinar" qué hacer en casos límite porque el árbol de decisiones está escrito explícitamente.

---

## Documento 3: El Plan Técnico (`plan.md`)

### Qué es y para qué sirve

El plan técnico traduce el QUÉ de la especificación funcional en el CÓMO. Aquí es donde entras en detalles técnicos: qué tecnologías usar, qué base de datos, cómo se comunican los servicios, qué esquemas de datos necesitas.

**Regla fundamental**: NO saltes del `spec.md` directamente al código. Eso es una receta para el desastre. El plan intermedio es lo que permite al agente (y a ti) validar que las decisiones técnicas tienen sentido antes de escribir una sola línea.

### Qué incluir

#### Stack tecnológico con versiones exactas

No digas "usar React". Di exactamente qué:

```markdown
## Stack tecnológico

- **Runtime**: Node.js 20 LTS
- **Framework**: Express.js 4.18
- **Lenguaje**: TypeScript 5.3 (modo estricto)
- **Base de datos**: PostgreSQL 16 (Prisma ORM 5.x)
- **Cache/Colas**: Redis 7.2
- **WebSockets**: Socket.io 4.7
- **Tests**: Vitest 1.x + Supertest
- **Despliegue**: Docker + Google Cloud Run
- **Mensajería**: Google Pub/Sub
```

#### Modelo de datos

```markdown
## Modelo de datos

### Tabla: notifications
| Campo | Tipo | Restricciones |
|-------|------|---------------|
| id | UUID | PK, auto-generado |
| user_id | UUID | FK → users.id, NOT NULL |
| order_id | UUID | FK → orders.id, NOT NULL |
| type | ENUM('status_change', 'delivery', 'promotion') | NOT NULL |
| title | VARCHAR(200) | NOT NULL |
| body | TEXT | NOT NULL |
| read | BOOLEAN | DEFAULT false |
| delivered | BOOLEAN | DEFAULT false |
| created_at | TIMESTAMP | DEFAULT now() |
| expires_at | TIMESTAMP | DEFAULT now() + 30 days |

**Invariantes:**
- `expires_at` siempre debe ser posterior a `created_at`
- `type` solo admite los valores del ENUM, sin extensiones
```

#### Contratos de API

Incluye los endpoints con sus esquemas de request/response. Esto es mucho más efectivo que describir la API con palabras:

```markdown
## API REST

### POST /api/notifications/send
**Descripción**: Enviar una notificación a un usuario

**Request:**
\```json
{
  "user_id": "uuid-del-usuario",
  "order_id": "uuid-del-pedido",
  "type": "status_change",
  "title": "Tu pedido ha sido enviado",
  "body": "El pedido #1234 está en camino"
}
\```

**Response 201 (Created):**
\```json
{
  "data": {
    "notification_id": "uuid-generado",
    "delivered": true,
    "delivered_at": "2025-01-15T10:30:00Z"
  }
}
\```

**Response 404:**
\```json
{
  "error": "User not found",
  "code": "USER_NOT_FOUND"
}
\```

**Response 422:**
\```json
{
  "error": "Invalid notification type",
  "code": "INVALID_TYPE",
  "valid_types": ["status_change", "delivery", "promotion"]
}
\```
```

#### Flujo de comunicación

Para agentes en la nube, define cómo se comunican los servicios:

```markdown
## Flujo de comunicación

1. El servicio de Pedidos actualiza el estado de un pedido en la BD
2. El servicio de Pedidos publica un evento en Google Pub/Sub
   (topic: `order-status-changed`)
3. El servicio de Notificaciones consume ese evento
4. El servicio de Notificaciones:
   a. Crea el registro en la tabla `notifications`
   b. Busca la conexión WebSocket activa del usuario
   c. Si está conectado → envía por WebSocket
   d. Si NO está conectado → marca como `delivered: false`
5. Cuando el usuario reconecta, se le envían las notificaciones pendientes
```

### Firmas de funciones: el contrato técnico

En el nivel más avanzado de SDD, la spec define la interfaz exacta que el código debe exponer. En vez de escribir un párrafo diciendo "necesito una función de login que reciba email y contraseña y devuelva una sesión", escribes la firma directa:

```python
def login(email: str, password: str) -> Session: ...
def logout(session_id: str) -> None: ...
```

O en TypeScript:

```typescript
interface NotificationService {
  send(payload: NotificationPayload): Promise<NotificationResult>;
  markAsRead(notificationId: string, userId: string): Promise<void>;
  getPending(userId: string): Promise<Notification[]>;
}
```

Esto establece los nombres de parámetros, tipos de entrada y tipos de retorno de forma inequívoca. El agente puede implementar la lógica interna como quiera, pero la interfaz (el contrato) queda bloqueada por la spec.

---

## Documento 4: Las Tareas (`tasks.md`)

### Qué es y para qué sirve

Las tareas son el último eslabón: el plan descompuesto en unidades de trabajo pequeñas, concretas y ordenadas. Cada tarea debería corresponder a un commit de código y no afectar a más de 3–5 archivos simultáneamente.

**¿Por qué importa esto?** Si le pides al agente "implementa todo el sistema de notificaciones", va a generar 2.000 líneas de código de golpe, probablemente con errores que se propagan en cadena. Si le pides "implementa la tarea T3: crear el esquema de la tabla notifications", genera 30 líneas, las verificas, y avanzas.

### Ejemplo de archivo de tareas

```markdown
## Tareas — Sistema de Notificaciones

### T1: Crear esquema de base de datos [Story: HU-1]
- **Archivos**: `prisma/schema.prisma`, `prisma/migrations/`
- **Acción**: Añadir modelo Notification con los campos definidos en plan.md
- **Verificación**:
  - [ ] `npx prisma migrate dev` ejecuta sin errores
  - [ ] El modelo tiene todos los campos de la tabla `notifications`
  - [ ] Las relaciones FK con `users` y `orders` son correctas

### T2: Crear servicio de notificaciones [Story: HU-1] [depende de T1]
- **Archivos**: `src/services/notification.service.ts`,
                `tests/services/notification.service.test.ts`
- **Acción**: Implementar `NotificationService` con métodos `send()`,
  `markAsRead()` y `getPending()` según las firmas del plan
- **Verificación**:
  - [ ] Tests unitarios pasan: `npm test -- notification.service`
  - [ ] Cobertura > 80% en el servicio

### T3: Crear endpoint REST [Story: HU-1] [depende de T2]
- **Archivos**: `src/routes/notification.routes.ts`,
                `tests/routes/notification.routes.test.ts`
- **Acción**: POST /api/notifications/send con validación Zod
- **Verificación**:
  - [ ] Test de integración con Supertest pasa
  - [ ] Response 201 con el formato especificado en plan.md
  - [ ] Response 404 y 422 para casos de error

### T4: Integrar WebSocket [Story: HU-1] [P] [depende de T2]
- **Archivos**: `src/websocket/notification.gateway.ts`,
                `tests/websocket/notification.gateway.test.ts`
- **Acción**: Configurar Socket.io para emitir notificaciones en tiempo real
- **Verificación**:
  - [ ] El usuario conectado recibe la notificación en < 5 segundos
  - [ ] Las notificaciones pendientes se entregan al reconectar

### T5: Consumidor Pub/Sub [Story: HU-1] [depende de T3, T4]
- **Archivos**: `src/consumers/order-status.consumer.ts`
- **Acción**: Suscribirse al topic `order-status-changed`
  y orquestar el flujo completo (BD → WebSocket)
- **Verificación**:
  - [ ] Test con evento simulado pasa end-to-end
  - [ ] Los reintentos funcionan (máximo 3)
```

Observa los marcadores:
- `[Story: HU-1]` vincula cada tarea con su historia de usuario
- `[depende de T1]` establece el orden de ejecución
- `[P]` indica que la tarea puede ejecutarse en paralelo con otras
- Cada tarea tiene archivos concretos y criterios de verificación medibles

---

## ¿Dónde encaja el código en las especificaciones?

Esta es una de las dudas más frecuentes. La respuesta corta: **las specs SÍ incluyen código, pero no el código de implementación**. Incluyen código para definir contratos, ejemplos de estilo y configuraciones.

### Dónde SÍ usar snippets de código

| Documento | Tipo de código | Ejemplo |
|-----------|---------------|---------|
| **constitution.md** | Código de referencia estilística | Un componente "perfecto" que muestre cómo se escriben las cosas en tu proyecto |
| **constitution.md** | Comandos ejecutables | `npm test -- --coverage`, `docker compose up -d` |
| **spec.md** | Payloads de ejemplo (entrada/salida) | JSONs de request/response |
| **spec.md** | Pseudocódigo para aceptación | "Si email no tiene @, retornar 422" |
| **plan.md** | Firmas de funciones | `async send(payload: NotificationPayload): Promise<Result>` |
| **plan.md** | Esquemas de datos | Definiciones OpenAPI, esquemas JSON, modelos Prisma |
| **plan.md** | Diagramas de flujo (Mermaid) | Secuencias de comunicación entre servicios |
| **tasks.md** | Rutas de archivos exactas | `src/services/notification.service.ts` |

### Dónde NO usar código

- **NUNCA** en spec.md: lógica interna de funciones, algoritmos completos, implementaciones de controladores
- **NUNCA** en constitution.md: implementaciones específicas de una feature

La regla: el código en las specs define **interfaces y contratos** (lo que se ve desde fuera), no **implementación** (cómo funciona por dentro).

---

## Especificaciones para agentes Cloud: lo que cambia

Cuando tu proyecto involucra agentes de comunicación en la nube, las especificaciones necesitan cubrir aspectos adicionales que no existen en un proyecto web tradicional.

### Definir los propios agentes

Si tu sistema incluye múltiples agentes de IA que colaboran, necesitas especificar cada uno como una entidad con identidad, rol y reglas:

```markdown
## Agente: @Notification-Manager

**Rol**: Gestionar el ciclo de vida completo de las notificaciones push
**Mandato**: "Garantizar que toda notificación relevante llegue al usuario
             correcto en menos de 5 segundos"

**Tono de comunicación**: Preciso y conciso. Reporta métricas
                          en cada interacción.

**Patrón de razonamiento**: Basado en objetivos (planifica secuencias
                            para alcanzar el resultado)

**Compuertas de aprobación humana**:
- Antes de modificar la topología de temas en Pub/Sub
- Cuando los fallos de entrega superen el 5% en una ventana de 1 hora
- Antes de escalar automáticamente a más de 10 instancias
```

### Protocolos de comunicación entre agentes

Para que varios agentes trabajen juntos sin conflictos, necesitas definir cómo se hablan:

```markdown
## Protocolos de comunicación inter-agente

- **Entre agentes locales (mismo IDE)**: Model Context Protocol (MCP)
  sobre JSON-RPC 2.0. El agente solicita el contrato específico
  del servidor antes de generar código.

- **Entre agentes remotos (nube)**: Agent-to-Agent Protocol (A2A)
  mediante "Agent Cards" que describen las capacidades de cada agente.

- **Formato de mensajes**: JSON con campos obligatorios:
  - `correlation_id`: UUID para rastreo distribuido
  - `sender`: identificador del agente emisor
  - `timestamp`: ISO 8601
  - `payload`: datos del mensaje
  - `schema_version`: versión del contrato de datos
```

### AsyncAPI: el estándar para comunicación asíncrona

Los agentes Cloud rara vez funcionan con simple petición/respuesta. Operan con eventos: "cuando pase X, haz Y". Para esto, OpenAPI (que describe APIs REST síncronas) no basta. Necesitas **AsyncAPI**, que hace lo mismo pero para eventos asíncronos (WebSockets, Kafka, MQTT, colas de mensajes...).

Un esquema AsyncAPI en tu spec define:

1. **Servidores**: URLs del broker de mensajería, protocolos, puertos, autenticación
2. **Canales**: Las "tuberías" por donde fluyen los datos (topics, colas). Cada canal especifica si tu aplicación envía (`send`) o recibe (`receive`)
3. **Mensajes**: La estructura exacta de los datos en tránsito — tipos, campos obligatorios, enumeraciones permitidas
4. **Errores**: Códigos de fallo que el agente puede encontrar y cómo reaccionar

Incluir un esquema AsyncAPI bien diseñado en tu spec es la forma más directa de asegurar que el agente genere código que conecte perfectamente con el resto de tu ecosistema de eventos.

### Gestión de memoria del agente

Un aspecto que se olvida frecuentemente: cómo gestiona el agente su propia memoria.

```markdown
## Gestión de memoria

### Memoria a corto plazo (sesión)
- Se almacena en Redis con TTL de 24 horas
- Contiene: historial de interacción actual, lista de tareas pendientes
- Se borra al finalizar la sesión del agente

### Memoria a largo plazo (aprendizaje)
- Las sesiones finalizadas se resumen automáticamente
  y se almacenan como embeddings en una base vectorial (Pinecone)
- El agente puede buscar experiencias pasadas similares
  antes de tomar decisiones sobre nuevas tareas
```

---

## Infraestructura como Código (IaC): especificar el entorno Cloud

Las exigencias de SDD no terminan en el código de la aplicación. Cuando instruyes a un agente para crear infraestructura Cloud (con Terraform, CloudFormation, etc.), la especificación es igual de importante o más, porque los errores de configuración pueden ser muy costosos (en seguridad y en dinero).

### Principios para especificar infraestructura

1. **Primero documentación oficial**: Exige que el agente lea la documentación actualizada del proveedor Cloud antes de generar recursos. Los parámetros de IAM, redes y almacenamiento cambian constantemente.

2. **Modularizar por componentes**: No pidas "despliega toda la infraestructura". Divide en stacks: red, base de datos, aplicación, monitorización. Cada stack tiene su propia spec.

3. **Empezar agnóstico, terminar específico**: En la spec de alto nivel, usa términos genéricos ("base de datos relacional", "almacenamiento de objetos"). Solo en el plan técnico introduces la nomenclatura del proveedor (`Cloud SQL`, `S3`, `REGION=us-central1`).

4. **Usar tablas para perfiles de recursos**:

```markdown
## Perfiles de servidor

| Perfil | vCPUs | RAM | Disco | Red | VLAN |
|--------|-------|-----|-------|-----|------|
| Small | 2 | 4 GB | 50 GB | 200 Mbps | 10 |
| Medium | 8 | 16 GB | 200 GB | 500 Mbps | 10 |
| Large | 16 | 64 GB | 500 GB | 800 Mbps | 20 |

El servicio de notificaciones en producción requiere perfil **Medium**.
El broker de mensajería requiere perfil **Large** con aislamiento en VLAN 20.
```

---

## El flujo de trabajo completo: paso a paso

Ahora que conoces los documentos, veamos cómo se usan en la práctica. El proceso es como una cadena de montaje con "compuertas" — solo avanzas a la siguiente fase cuando has validado la anterior.

### Fase 1: Specify (definir el problema)

1. **Tú escribes** (o le pides al agente que genere un borrador) el `spec.md`
2. **Revisas** el borrador: ¿faltan casos? ¿hay ambigüedades? ¿hay "alucinaciones" (cosas que inventó el agente)?
3. **Resuelves** los marcadores `[NECESITA CLARIFICACIÓN]` — máximo 3 por spec
4. **Apruebas** el documento

**Herramientas**: `/speckit.specify`, modo Plan de Claude Code, Kiro

### Fase 2: Plan (diseño técnico)

1. Con la spec aprobada, **generas** el `plan.md`
2. El agente propone arquitectura, stack, esquemas de datos, contratos API
3. **Validas** que el diseño respeta tu constitución y tu infraestructura Cloud
4. **Apruebas** el plan

**Herramientas**: `/speckit.plan`, `/plan` en OpenSpec

### Fase 3: Tasks (atomizar el trabajo)

1. El plan aprobado se **descompone** en tareas atómicas
2. Cada tarea tiene archivos concretos, dependencias y criterios de verificación
3. **Revisas** que ninguna tarea sea demasiado amplia (regla: ≤ 5 archivos por tarea)
4. **Apruebas** la lista de tareas

**Herramientas**: `/speckit.tasks`, `/tasks` en OpenSpec

### Fase 4: Implement (código y verificación)

1. El agente **ejecuta** las tareas una por una
2. Para cada tarea: escribe tests → hace que fallen → implementa → hace que pasen
3. **Revisas** cada cambio como un pull request enfocado
4. Los tests automáticos **verifican** que el código cumple la spec original

**Herramientas**: `/speckit.implement`, Claude Code, Cursor

**Importante**: NO trates al agente como un buscador web al que le escribes un prompt y esperas un sistema funcional. En SDD, el agente avanza solo cuando tú abres la compuerta validando la fase anterior.

---

## Herramientas del ecosistema SDD

### Las principales

| Herramienta | Qué es | Para qué la usas |
|-------------|--------|-------------------|
| **GitHub Spec Kit** | Framework open-source (69k+ ⭐) | Bootstrappear la estructura de specs y ejecutar el flujo completo con comandos slash |
| **AWS Kiro** | IDE (fork de VS Code) | Genera requisitos, diseño y tareas desde lenguaje natural. De gratis a $39/mes |
| **Tessl** | Framework + registro de specs | Nivel spec-as-source con mapeo 1:1 entre specs y código |
| **OpenSpec** | Framework ligero | Comandos slash sin gates rígidos entre fases |
| **SPARC/Claude-Flow** | Orquestación multi-agente | 17 modos especializados (Arquitecto, TDD, Security...) |

### Herramientas de soporte

| Herramienta | Para qué |
|-------------|----------|
| **ContextPilot** | Auto-genera `.cursorrules`, `CLAUDE.md` y `copilot-instructions.md` desde tu codebase |
| **rulesync** | Escribe reglas una vez, genera para todos los agentes |
| **Repomix** | Empaqueta tu repo entero en un archivo que la IA puede leer |
| **AGENTS.md** | Estándar emergente para instrucciones universales a agentes (como README pero para la IA) |
| **mdschema** | Valida que tus specs Markdown tengan las secciones obligatorias |

---

## Validación y detección de "deriva"

SDD solo funciona si puedes verificar automáticamente que el código coincide con la spec. Sin esto, la spec se convierte en documentación muerta como cualquier otra.

### El problema de la "deriva" de specs

La "deriva" (spec drift) ocurre cuando alguien modifica el código sin actualizar la spec. Es el fallo más insidioso de SDD porque todo parece funcionar hasta que otro agente (o tú mismo dentro de 3 meses) lee la spec obsoleta y genera código incompatible.

### Cómo prevenirla

1. **Pruebas de contrato**: Herramientas que verifican que tu API real coincide con la definición OpenAPI/AsyncAPI de tu plan.

2. **Validación de estructura de specs**: Usa `mdschema` para definir un esquema YAML que obligue a que cada `spec.md` tenga secciones de "Criterios de Aceptación" y que cada tarea en `tasks.md` esté vinculada a un requisito.

3. **Trazabilidad con etiquetas de test**: Frameworks como Tessl permiten incluir `[@test] ../tests/auth/test_login.py` directamente en la spec, creando un vínculo explícito entre cada requisito y su test.

4. **CI/CD con gates de spec**: Configura tu pipeline para que rechace PRs donde el código cambie pero la spec no se haya actualizado.

---

## Los 8 errores más comunes (y cómo evitarlos)

### 1. Dejar que la IA escriba la spec sin revisarla

Cuando le pides al agente que genere tu spec, tiende a producir documentos inflados con suposiciones innecesarias. Un equipo reportó que Spec Kit generó "muros de texto" de 1.300 líneas para producir 689 líneas de código. Usa las specs generadas como punto de partida, no como resultado final.

### 2. Escribir specs vagas

"Hazme un sistema de notificaciones moderno y escalable" no le dice nada útil al agente. Necesita saber: qué tipo de notificaciones, por qué canal, con qué latencia máxima, qué pasa cuando fallan, y qué NO debe incluir.

### 3. Hacer un único documento gigante

Si metes toda la especificación en un solo archivo de 50 páginas, el agente perderá información crítica. Divide en constitution → spec → plan → tasks. Cada documento tiene un propósito y un tamaño manejable.

### 4. Saltar del spec.md directamente al código

Sin el paso intermedio del plan técnico, el agente tomará decisiones arquitectónicas que podrían ser incorrectas. El plan es donde validas esas decisiones antes de que se conviertan en código.

### 5. Crear tareas demasiado grandes

"Implementar el servicio de notificaciones completo" es una tarea terrible. El agente generará miles de líneas de una vez, con errores que se propagan. Mejor: "Crear el esquema de base de datos para notifications" — 30 líneas, verificables.

### 6. No definir qué está prohibido

Los modelos de IA son creativos por defecto. Si no les dices "NUNCA pongas lógica de negocio en los routers", lo harán tarde o temprano. Los anti-patrones explícitos son tan importantes como las instrucciones positivas.

### 7. Tratar SDD como waterfall

La crítica más habitual: "esto es waterfall disfrazado de Markdown". La diferencia clave es la velocidad del ciclo. En waterfall, escribías specs durante meses y luego descubrías los errores. En SDD, escribes la spec, el agente implementa en minutos, ves qué falla, actualizas la spec y regeneras. El ciclo de feedback pasa de meses a minutos.

### 8. No mantener la spec actualizada

Si cambias el código pero no actualizas la spec, la próxima vez que el agente lea la spec generará código basado en información obsoleta. La spec es un documento vivo. Si le quitas la vida, se convierte en ficción.

---

## Plantilla rápida para empezar hoy

Si quieres adoptar SDD sin complicarte la vida, empieza con esto:

### Paso 1 (5 minutos): Crea un AGENTS.md mínimo

```markdown
# Proyecto: Mi Agente Cloud

## Comandos
- Tests: `npm test`
- Build: `npm run build`
- Dev: `npm run dev`

## Estructura
src/routes/    → Rutas (sin lógica de negocio)
src/services/  → Lógica de negocio
src/clients/   → Acceso a datos y APIs
tests/         → Tests (espejo de src/)

## Estilo
- TypeScript estricto
- Funciones async/await
- Exportaciones nombradas

## Prohibido
🚫 Lógica de negocio en routes
🚫 Secretos en código
🚫 Borrar tests sin resolver
```

### Paso 2 (10 minutos): Escribe tu primera spec

Elige una funcionalidad, escribe las historias de usuario y los criterios de aceptación. No te preocupes por la perfección — una spec del 80% es infinitamente mejor que ninguna spec.

### Paso 3 (5 minutos): Genera el plan

Pide al agente que genere el plan técnico a partir de tu spec. Revísalo, ajústalo, apruébalo.

### Paso 4: Implementa con tareas atómicas

Deja que el agente descomponga el plan en tareas y ejecútalas una a una.

**El ROI de SDD escala directamente con la complejidad del proyecto.** Para un botón, es innecesario. Para un sistema de pagos distribuido, es transformador. Empieza pequeño, escala gradualmente.

---

## Conclusión

SDD no es documentación extra. Es una forma diferente de programar donde tú describes con precisión lo que quieres y la IA lo construye. Los documentos de especificación son tu herramienta principal — no el chat, no los prompts improvisados.

Lo que necesitas recordar:

- **constitution.md** = las reglas sagradas que nunca cambian
- **spec.md** = QUÉ construir y POR QUÉ (sin detalles técnicos)
- **plan.md** = CÓMO construirlo (stack, esquemas, contratos API)
- **tasks.md** = los pasos concretos, uno por uno

Los tres mandamientos de un buen spec:
1. **Sé específico**: "menos de 5 segundos" mejor que "rápido"
2. **Di lo que NO hay que hacer**: los anti-patrones son tan importantes como las instrucciones
3. **Incluye ejemplos de código**: un snippet vale más que un párrafo

El futuro del desarrollo con IA no es escribir código más rápido. Es escribir mejores especificaciones. La claridad se ha convertido en la habilidad más valiosa de un programador.
