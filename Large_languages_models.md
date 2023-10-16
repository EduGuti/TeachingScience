# Introducción

Por ahora sólo voy a usar este fichero para ir compartiendo notas cortas sobre este tema. Espero que más adelante yo pueda escribir algo bien redactado sobre este tema.

# Introducción al "prompting".
En el contexto de LLMs (modelos de lenguaje grandes, o modelos grandes de lenguaje), un "prompt" es el texto que el correspondiente modelo recibe para generar una respuesta. A groso modo, puede ser pensado como el texto que 
el usuario escribe y envía, por ejemplo en ChatGPT, pero en realidad la aplicación suele añadir más texto, que no es mostrado al usuario. Este tema ha cogido tanta popularidad, que actualmente (año 2023) está de moda la 
llamada "ingeniería de prompting".

Como **introducción a este tema**, recomiendo el **vídeo** [Tutorial de ingeniería rápida: respuestas maestras de ChatGPT y LLM](https://www.youtube.com/watch?v=_ZvnD73m40o).

**Ejemplo** creado por mí el día 2023-10-16 **con *ChatGPT Plus*** (en España, con GPT-4 y seleccionada la búsqueda con Bing (que a ese día aún está como herramienta "beta"); actualmente, *ChatGPT Plus* cuesta 20 $ por mes sin impuestos y 24,40 $ por mes con impuestos, lo cual vienen a ser entre 22 y 23 €/mes):
**Ejemplo de estudio de clasificación biológica**: [Eres un biólogo especializado en clasificación de seres vivos. Conoces las características de cada rango jerárquico necesario para clasificar cada especie, desde el superior hasta la especie o subespecie. Estás familiarizado con diferentes tipos de clasificación, pero puedes consultar Internet para ampliar o actualizar tus conocimientos. Muestra un listado de todos los rangos de la siguiente especie, poniendo junto a cada rango (nivel jerárquico) las características de ese rango y sólo de ese rango, es decir, sin repetirlas en varios/otros rangos o cualquier otra descripción. Intenta ser lo más específico posible; es decir, pon la mayor cantidad de características posible, y para eso elige el tipo de clasificación más completo que conozcas, y al comienzo de tu respuesta indica cuál has elegido. Como tienes que buscar mucho en Internet. Muestra 2 rangos de cada vez. ESPECIE: Plexippus paykulli](https://chat.openai.com/share/b04b695a-c010-499a-8dc6-3d802d64f9ee)

Y para hacer eso **con imágenes** (característica/funcionalidad nueva en ChatGPT Plus; desde el día 2023-09-23 en EEUU y desde el día 2023-10-14 (al menos yo la vi ese día) en España, y supongo que en toda Europa), hay un estupendo paper/artículo con **ejemplos de** creación de **prompts que contienen texto e imágenes**:
 - [Página del **paper** en Arxiv](https://arxiv.org/abs/2309.17421)
 - **Vídeo**, sobre ese paper, del canal de YouTube [1littlecoder](https://www.youtube.com/@1littlecoder): [GPT-4 Vision PROMPT ENGINEERING Tutorial](https://www.youtube.com/watch?v=dtKBAST12O0)
