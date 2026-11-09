# coder_ai_eng_course

Cliente asíncrono para usar modelos de **OpenAI** y **Anthropic** con una misma interfaz. Para elegir el proveedor solo hay que cambiar un parámetro, sin modificar el resto del código.

## Requisitos

- Python 3.14. En Windows se ejecuta con el lanzador `py`.
- Una API key de OpenAI y/o de Anthropic.

## Instalación

Desde la raíz del proyecto, en PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install openai anthropic pydantic python-dotenv
```

Opcionalmente, podés activar el entorno virtual para usar `python` directamente:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Variables de entorno

Copiá la plantilla y completá tus claves:

```powershell
Copy-Item .env.example .env
```

| Variable | Para qué sirve |
|---|---|
| `LLM_PROVIDER` | Proveedor que usa `main.py`: `openai` o `anthropic` |
| `LLM_MODEL` | Modelo que usa `main.py`, por ejemplo `claude-opus-5-5` |
| `OPENAI_API_KEY` | Necesaria si usás el proveedor `openai` |
| `ANTHROPIC_API_KEY` | Necesaria si usás el proveedor `anthropic` |

Solo hace falta la clave del proveedor que vayas a usar. El archivo `.env` está en `.gitignore`, así que nunca se sube al repositorio.

`main.py` carga el `.env` con `load_dotenv()`. `AsyncLLMManager` no lo carga: solo lee las variables de entorno, sin importar de dónde vengan.

`main.py` muestra `[config error] ...` y termina sin traceback en estos casos:
- `LLM_PROVIDER` o `LLM_MODEL` faltan o tienen un valor inválido. Los rechaza `LLMConfig`.
- La clave del proveedor elegido no está definida o está vacía. El manager lanza un `ValueError` al crearse, antes de llamar a la API.

## Ejecutar el script de prueba

```powershell
.\.venv\Scripts\python.exe main.py
```

`main.py` hace la pregunta *"¿Qué es la entropía?"* dos veces:

1. **Modo normal** (`chat`): imprime la respuesta completa, los tokens usados y el motivo de corte (`stop`).
2. **Streaming** (`generate`): imprime la respuesta a medida que llega.

Para cambiar de proveedor o de modelo, editá `LLM_PROVIDER` y `LLM_MODEL` en el `.env`:

```
LLM_PROVIDER=openai
LLM_MODEL=gpt-5
```

## Uso desde código

Cargar el `.env` es tarea de la aplicación, no del manager:

```python
from dotenv import load_dotenv

from llm import AsyncLLMManager
from schemas import ChatMessage, LLMConfig, LLMError

load_dotenv()
config = LLMConfig(provider="anthropic", model="claude-opus-5-5")

async with AsyncLLMManager(config) as llm:
    result = await llm.chat([ChatMessage(role="user", content="Hola")])
    if isinstance(result, LLMError):
        print(result.kind, result.message)
    else:
        print(result.content)
```

`async with` cierra las conexiones HTTP al salir del bloque.

### Parámetros de generación

`GenerationParams` (en `schemas.py`) valida los parámetros antes de llamar a la API:

```python
params = GenerationParams(max_tokens=4000, temperature=0.3)
await llm.chat(messages, params)
```

- `max_tokens` funciona en los dos proveedores. Si la respuesta se corta, vas a ver `stop: max_tokens`. En los modelos de Claude el razonamiento interno también consume este límite, así que conviene un valor holgado.
- `temperature` y `top_p` se aplican solo en OpenAI. Los modelos actuales de Claude no los aceptan, así que `AnthropicClient` los ignora y registra un warning.

## Manejo de errores

Los SDKs reintentan solos los errores temporales: límite de uso (429), errores del servidor (5xx) y errores de conexión. Entre intentos esperan cada vez más y respetan el tiempo que indique la API. Para cambiar la cantidad de reintentos:

```python
AsyncLLMManager(LLMConfig(provider="openai", model="...", max_retries=5))
```

Si después de los reintentos el error persiste, el manager no lanza la excepción: devuelve un `LLMError`.

- En `chat()`, el `LLMError` llega en lugar de la respuesta.
- En `generate()`, llega como último elemento del stream, después del texto parcial que se haya recibido. Esto incluye los cortes de conexión en medio del stream, con los dos proveedores: llegan como `kind="connection"`.

| `kind` | Causa típica | `retryable` |
|---|---|---|
| `auth` | API key inválida o sin permisos (401/403) | No |
| `rate_limit` | Límite de uso o cuota agotada (429) | Sí |
| `connection` | Sin conexión o timeout | Sí |
| `api` | Otro error de la API (400, 404, 5xx…) | Solo los 5xx |

Los errores de programación, como pasar un `ChatMessage` inválido, no se capturan: siguen lanzando su excepción para que los veas.
