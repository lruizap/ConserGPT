# ConserGPT

ConserGPT es una interfaz de conversación que utiliza el modelo Command R+ de OpenAI para proporcionar respuestas a preguntas relacionadas con el sistema educativo andaluz. Esta interfaz está diseñada para ayudar a responder preguntas específicas sobre programas educativos, normativas y políticas en Andalucía.

## Características

- Interfaz de conversación basada en Gradio.
- Utiliza el modelo Command R+ de OpenAI para generar respuestas.
- Proporciona información sobre programas educativos, normativas y políticas en Andalucía.
- Incluye un archivo .env para la configuración de las variables de entorno.

## Inicialización del Proyecto

1. Clona este repositorio en tu máquina local:

```bash
git clone https://github.com/lruizap/ConserGPT.git
```

2. Crea un entorno virtual para el proyecto:

```bash
cd ConserGPT
python -m venv venv
```

3. Activa el entorno virtual:

- En Windows:

```bash
venv\Scripts\activate
```

- En macOS y Linux:

```bash
source venv/bin/activate
```

4. Instala las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

5. Crea un archivo `.env` en el directorio raíz del proyecto y añade las siguientes variables de entorno:

```plaintext
OPENROUTER_API_KEY=your_openrouter_api_key
LANGFUSE_PRIVATE_API_KEY=your_langfuse_private_api_key
LANGFUSE_PUBLIC_API_KEY=your_langfuse_public_api_key
```

Asegúrate de reemplazar `your_openrouter_api_key`, `your_langfuse_private_api_key` y `your_langfuse_public_api_key` con tus propias claves API.

6. Ejecuta la aplicación:

```bash
python app.py
```

7. Accede a la aplicación en tu navegador web utilizando la URL proporcionada en la terminal.

## Uso

Una vez que la aplicación esté en funcionamiento, podrás seleccionar una pregunta predefinida sobre el sistema educativo andaluz y obtener una respuesta generada por el modelo Command R+ de OpenAI.
