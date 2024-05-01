# LangChain

## Introducción

Cuando conocí *LangChain*, me gustó porque me permitía, con poco código, empezar a usar LLMs locales, cuando yo aún no sabía mucho sobre ese tema. Y luego, al conocerlo un poco mejor, me gustó toda la parte de RAG, ya que tenía implementada la carga de muchos tipos de documentos, e incluso facilitaba "scraping" de varios sitios webes. Pero a día 2024-05-01, y desde hace ya cierto tiempo, me parece difícil de usar para muchos casos de uso, y ya no digamos si quiero entender bien lo que internamente está haciendo, o añadir alguna funcionalidad, ya que su código me resulta tremendamente confuso (no entiendo el por qué de muchas cosas, incluyendo el motivo por el que muchas clases existen, ni la organización de sus directorios; no digo que no tengan sentido (no lo sé), pero no es nada intuitivo para el que no ha estado en su creación desde el principio de todo, e incluso sabiendo bastante sobre LLMs). Por todo eso, quiero ir documentando aquí (poco a poco; WIP, jeje) ese código.

Lo primero de todo, hay que tener en cuenta que ese código (no todo, pero sí la mayor parte, y todo lo más importante a la hora de usarlo) está dividido en 2 grandes grupos, que son los siguientes.
 - [langchain_core](https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core)
 - [langchain_community](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community)

Voy a empezar mayormente por el 2º (con algunas excepciones); como he dicho, esta documentación es WIP, así que es posiblel que lo acabe cambiando, pero por ahora quiero empezar por ahí porque esa parte es la más próxima al usuario final, y así por lo menos, al mirar "recetas" (tutoriales de estos cutres (porque no explican nada) y que tanto abundan), con esto será posible entender un poco de eso. Y voy a empezar poniendo un listado de ficheros de ese código (no todos, porque son muchísimos y muchos de ellos sólo son necesarios en ciertos casos de uso muy concretos/específicos), y en el orden en el que me parece más didáctico; en concreto, voy a empezar por los modelos de lenguaje, ya que es el origen, y motivación, de este proyecto, y por los "chats", ya que también es una parte importante de todo esto (ya que la gran popularidad de los LLMs ha sido consecuencia de la aparición de (ese "milagro" llamado) 'ChatGPT'). Por ahora sólo los listo, y poco a poco (a lo largo de varios días o semanas) los voy a ir comentando (o eso espero).
 - [libs/core/langchain_core/language_models/__init__.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/__init__.py)
 - [libs/core/langchain_core/language_models/base.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/base.py)
 - [libs/core/langchain_core/language_models/llms.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/llms.py)
 - [libs/core/langchain_core/language_models/fake.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/fake.py)
 - [libs/community/langchain_community/llms/fake.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/fake.py)
 - [libs/community/langchain_community/llms/human.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/human.py)
 - [libs/core/langchain_core/language_models/chat_models.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/chat_models.py)
 - [libs/core/langchain_core/language_models/fake_chat_models.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/fake_chat_models.py)
 - [libs/community/langchain_community/chat_models/fake.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_models/fake.py)
 - [libs/community/langchain_community/chat_models/human.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_models/human.py)
 - [libs/community/langchain_community/llms/openai.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/openai.py)
 - [libs/community/langchain_community/llms/ollama.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/ollama.py)
 - [libs/community/langchain_community/llms/ctransformers.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/ctransformers.py)
 - [libs/community/langchain_community/llms/gpt4all.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/gpt4all.py)
 - [libs/community/langchain_community/llms/huggingface_hub.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/huggingface_hub.py)
 - [libs/community/langchain_community/llms/huggingface_pipeline.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/huggingface_pipeline.py)
 - [langchain_community/llms/huggingface_text_gen_inference.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/huggingface_text_gen_inference.py)
 - [libs/community/langchain_community/llms/huggingface_endpoint.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/huggingface_endpoint.py)
 - [libs/community/langchain_community/llms/llamacpp.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/llamacpp.py)
 - [libs/community/langchain_community/llms/llamafile.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/llamafile.py)
 - [libs/community/langchain_community/llms/loading.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/loading.py)
 - [libs/community/langchain_community/llms/modal.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/modal.py)
 - [libs/community/langchain_community/chat_models/openai.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_models/openai.py)
 - [libs/community/langchain_community/chat_models/ollama.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_models/ollama.py)
 - [libs/community/langchain_community/chat_models/huggingface.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_models/huggingface.py)
 - [libs/community/langchain_community/chat_models/gpt_router.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_models/gpt_router.py)
 - [libs/community/langchain_community/chat_models/meta.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_models/meta.py)
 - [libs/community/langchain_community/chat_message_histories/file.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_message_histories/file.py)
 - [libs/community/langchain_community/chat_message_histories/mongodb.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_message_histories/mongodb.py)
 - [libs/community/langchain_community/chat_message_histories/postgres.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_message_histories/postgres.py)
 - [libs/community/langchain_community/chat_message_histories/neo4j.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_message_histories/neo4j.py)
 - [libs/community/langchain_community/chat_message_histories/redis.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_message_histories/redis.py)
 - [libs/community/langchain_community/chat_message_histories/sql.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_message_histories/sql.py)
 - [libs/community/langchain_community/chat_loaders/utils.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_loaders/utils.py)
 - [libs/community/langchain_community/chat_loaders/gmail.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_loaders/gmail.py)
 - [libs/community/langchain_community/chat_loaders/telegram.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_loaders/telegram.py)
 - [libs/community/langchain_community/chat_loaders/whatsapp.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_loaders/whatsapp.py)
 - [libs/community/langchain_community/agent_toolkits/sql/base.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/agent_toolkits/sql/base.py)
 - [libs/community/langchain_community/agent_toolkits/sql/prompt.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/agent_toolkits/sql/prompt.py)
 - [libs/community/langchain_community/agent_toolkits/sql/toolkit.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/agent_toolkits/sql/toolkit.py)
 - [libs/community/langchain_community/agent_toolkits/playwright/toolkit.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/agent_toolkits/playwright/toolkit.py)
 - [libs/community/langchain_community/cache.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/cache.py)
 - [libs/community/langchain_community/utilities/wikipedia.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/wikipedia.py)
 - [libs/community/langchain_community/utilities/wikidata.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/wikidata.py)
 - [libs/community/langchain_community/utilities/stackexchange.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/stackexchange.py)
 - [libs/community/langchain_community/utilities/sql_database.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/sql_database.py)
 - [libs/community/langchain_community/utilities/requests.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/requests.py)
 - [libs/community/langchain_community/utilities/reddit_search.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/reddit_search.py)
 - [libs/community/langchain_community/utilities/python.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/python.py)
 - [libs/community/langchain_community/utilities/merriam_webster.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/merriam_webster.py)
 - [libs/community/langchain_community/utilities/arxiv.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/arxiv.py)
 - [libs/community/langchain_community/utilities/bibtex.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/bibtex.py)
 - [libs/community/langchain_community/utilities/duckduckgo_search.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/duckduckgo_search.py)
 - [libs/community/langchain_community/utilities/github.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/github.py)
 - [libs/community/langchain_community/utilities/openweathermap.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/openweathermap.py)
 - [libs/community/langchain_community/utilities/pubmed.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/pubmed.py)
 - [libs/community/langchain_community/tools/wikipedia/tool.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/wikipedia/tool.py)
 - [libs/community/langchain_community/tools/wikidata/tool.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/wikidata/tool.py)
 - [libs/community/langchain_community/tools/sql_database/prompt.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/sql_database/prompt.py)
 - [libs/community/langchain_community/tools/sql_database/tool.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/sql_database/tool.py)
 - [libs/community/langchain_community/tools/human/tool.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/human/tool.py)
 - [libs/community/langchain_community/tools/ddg_search/tool.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/ddg_search/tool.py)
 - [libs/community/langchain_community/tools/json/tool.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/json/tool.py)
 - [libs/community/langchain_community/tools/requests/tool.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/requests/tool.py)
 - [libs/community/langchain_community/tools/playwright](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/tools/playwright) (en este directorio a día 2024-05-01 hay 10 ficheros)
 - [libs/community/langchain_community/document_loaders/blob_loaders](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/document_loaders/blob_loaders) (4 ficheros)
 - [libs/community/langchain_community/document_loaders/parsers](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/document_loaders/parsers) (11 ficheros y 2 subdirectorios)
 - [libs/community/langchain_community/document_loaders/parsers/html/bs4.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_loaders/parsers/html/bs4.py)
 - [libs/community/langchain_community/document_loaders/wikipedia.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_loaders/wikipedia.py)
 - [libs/community/langchain_community/document_loaders/sql_database.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_loaders/sql_database.py)
 - [libs/community/langchain_community/document_loaders/web_base.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_loaders/web_base.py)
 - [libs/community/langchain_community/document_loaders/sql_database.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_loaders/sql_database.py)
 - [libs/community/langchain_community/document_transformers/__init__.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_transformers/__init__.py)
 - [libs/community/langchain_community/document_transformers/beautiful_soup_transformer.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_transformers/beautiful_soup_transformer.py)
 - [libs/community/langchain_community/document_transformers/html2text.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_transformers/html2text.py)
 - [libs/community/langchain_community/embeddings/fake.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/embeddings/fake.py)
 - [libs/community/langchain_community/embeddings/openai.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/embeddings/openai.py)
 - [libs/community/langchain_community/embeddings/ollama.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/embeddings/ollama.py)
 - [libs/community/langchain_community/embeddings/text2vec.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/embeddings/text2vec.py)
 - [libs/community/langchain_community/embeddings/tensorflow_hub.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/embeddings/tensorflow_hub.py)
 - [libs/community/langchain_community/embeddings/huggingface.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/embeddings/huggingface.py)
 - [libs/community/langchain_community/embeddings/huggingface_hub.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/embeddings/huggingface_hub.py)
 - [libs/community/langchain_community/embeddings/llamacpp.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/embeddings/llamacpp.py)
 - [libs/community/langchain_community/embeddings/llamafile.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/embeddings/llamafile.py)
 - [libs/community/langchain_community/retrievers/knn.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/retrievers/knn.py)
 - [libs/community/langchain_community/retrievers/svm.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/retrievers/svm.py)
 - [libs/community/langchain_community/retrievers/llama_index.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/retrievers/llama_index.py)
 - [libs/community/langchain_community/retrievers/wikipedia.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/retrievers/wikipedia.py)
 - [libs/community/langchain_community/retrievers/arxiv.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/retrievers/arxiv.py)
 - [libs/community/langchain_community/retrievers/pubmed.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/retrievers/pubmed.py)
 - [libs/community/langchain_community/utils](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/utils) (a día 2024-05-01 contiene 6 ficheros)
 - [libs/community/langchain_community/output_parsers](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/output_parsers) (a día 2024-05-01 contiene 3 ficheros)
 - [libs/community/langchain_community/document_compressors](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/document_compressors) (a día 2024-05-01 contiene 4 ficheros)
 - [libs/community/langchain_community/docstore](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/docstore) (a día 2024-05-01 contiene 6 ficheros)
 - [libs/community/langchain_community/docstore/__init__.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/docstore/__init__.py)
 - [libs/community/langchain_community/docstore/base.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/docstore/base.py)
 - [libs/community/langchain_community/docstore/in_memory.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/docstore/in_memory.py)
 - [libs/community/langchain_community/docstore/document.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/docstore/document.py) (sólo tiene 2 líneas de código)
 - [libs/community/langchain_community/docstore/wikipedia.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/docstore/wikipedia.py)
 - [libs/community/langchain_community/cross_encoders/__init__.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/cross_encoders/__init__.py)
 - [libs/community/langchain_community/cross_encoders/base.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/cross_encoders/base.py)
 - [libs/community/langchain_community/cross_encoders/fake.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/cross_encoders/fake.py)
 - [libs/community/langchain_community/cross_encoders/huggingface.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/cross_encoders/huggingface.py)
 - [libs/community/langchain_community/callbacks/human.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/callbacks/human.py)
 - [libs/community/langchain_community/callbacks/manager.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/callbacks/manager.py)
 - [libs/community/langchain_community/callbacks/utils.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/callbacks/utils.py)
 - [libs/community/langchain_community/adapters/__init__.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/adapters/__init__.py)
 - [libs/community/langchain_community/adapters/openai.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/adapters/openai.py)
 - [libs/community/langchain_community/__init__.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/__init__.py)
 - [libs/community/langchain_community/cache.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/cache.py)
 - [libs/community/langchain_community/indexes/__init__.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/indexes/__init__.py)
 - [libs/community/langchain_community/indexes/base.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/indexes/base.py)
 - [libs/community/langchain_community/indexes/_document_manager.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/indexes/_document_manager.py)
 - [libs/community/langchain_community/indexes/_sql_record_manager.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/indexes/_sql_record_manager.py)
 - [libs/community/langchain_community/example_selectors/__init__.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/example_selectors/__init__.py)
 - [libs/community/langchain_community/example_selectors/ngram_overlap.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/example_selectors/ngram_overlap.py)
 - [libs/community/langchain_community/graphs](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/graphs) (a día 2024-05-01 tiene 18 ficheros)
 - [libs/community/langchain_community/vectorstores](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/vectorstores) (desde el día 2024-02-01 ya tengo una sección sobre este tema, y es la siguiente en esta página web)

## Uso de vectores de características

A día 2024-02-01, del [código fuente de la gestión, de sistemas de almacenamiento de vectores, de *LangChain*](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/vectorstores), he mirado el contenido de los siguientes ficheros (de los 76 que hay ahí, si no he contado mal), los cuales a continuación listo en orden desde el más simple hasta el más complejo (en igualdad de complejidad, al menos en cuanto a cantidad de funcionalidades y de manera aproximada, pongo delante los que usan extensiones de otras bases de datos más generales y populares, como por ejemplo *SQLite*), y luego iré creando subsecciones para comentarlos.
 - [sklearn.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/sklearn.py)
 - [faiss.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/faiss.py)
 - [sqlitevss.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/sqlitevss.py)
 - [pgvector.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/pgvector.py)
 - [chroma.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/chroma.py)
 - [qdrant.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/qdrant.py)

### Con FAISS (de Meta AI)

A continuación voy a poner un esquema de las principales clases, y métodos, de un ejemplo de caso de uso de *LangChain* con *FAISS*.
El código está principalmente en el ficheor [faiss.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/faiss.py), pero aquí también voy a poner a poner las dependencias 
que me parezcan más relevantes para entender esa implementación.

Para usar eso, hay que empezar ejecutando lo siguiente:
<code lang="Python">from langchain.vectorstores import FAISS</code>

Existe la posibilidad de crear una nueva base de datos sólo en memoria, pero, dado que suele ser más útil guardarla en disco para poderla volver a usar, voy a considerar este segundo caso. En este caso, la definición 
de la base de datos va a depender de si es creada desde cero, o de si vamos a añadir más vectores a una ya existente en disco. En este último caso (si ya existe), primero hay que leerla desde disco, luego añadir los
nuevos textos y vectores, y finalmente sobreescribirla (obviamente también existe la posibilidad de crear nuevos ficheros, pero eso sería redundante; existe la posibilidad de dividirla en varios ficheros, por motivos
de eficiencia, pero eso es un método más avanzado y que por ahora no voy a describir aquí).

<pre>
  db_name = "texts_db"
  vectordb_exist = os.path.exists(db_name+".faiss")
  db = None
  texts = ["the dog is black", "the house is yellow"]
  if not vectordb_exist:
      db = FAISS.from_texts(texts, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
  else:
      db = FAISS.load_local(".", HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'), db_name)
      db.add_texts(texts)
  # En ambos casos, hay que guardar la base de datos a disco.
  db.save_local(".", db_name)
</pre>

A continuación pongo un esquema (no completo, sino sólo de los métodos que me parecen más relevantes) con las dependencias de esas 4 funciones usadas para crear ese almacenamiento de vectores.
Aquí no pongo la descripción, de cada método/clase y que está en el código fuente, para que este esquema ocupe menos y porque es posible verlo siguiendo el correspondiente enlace.
AVISO: Los enlaces indican la línea en la que el correspondiente métido está definido, en el código fuente de *LangChain*, a día 2024-01-30, pero puede ser que cambien eso y yo no voy a estar muy pendiente de actualizar esto.
WIP:
 - [from_texts](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/faiss.py#L933)
   <pre>
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> FAISS:
        embeddings = embedding.embed_documents(texts)
        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )
   </pre>
 - [load_local](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/faiss.py#L1085)
   <pre>
    @classmethod
    def load_local(
        cls,
        folder_path: str,
        embeddings: Embeddings,
        index_name: str = "index",
        **kwargs: Any,
    ) -> FAISS:
        path = Path(folder_path)
        # load index separately since it is not picklable
        faiss = dependable_faiss_import()
        index = faiss.read_index(
            str(path / "{index_name}.faiss".format(index_name=index_name))
        )

        # load docstore and index_to_docstore_id
        with open(path / "{index_name}.pkl".format(index_name=index_name), "rb") as f:
            docstore, index_to_docstore_id = pickle.load(f)
        return cls(embeddings, index, docstore, index_to_docstore_id, **kwargs)
   </pre>
 - [add_texts](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/faiss.py#L207)
   <pre>
     def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        texts = list(texts)
        embeddings = self._embed_documents(texts)
        return self.__add(texts, embeddings, metadatas=metadatas, ids=ids)
   </pre>
 - [save_local](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/faiss.py#L1063)
   <pre>
     def save_local(self, folder_path: str, index_name: str = "index") -> None:
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)

        # save index separately since it is not picklable
        faiss = dependable_faiss_import()
        faiss.write_index(
            self.index, str(path / "{index_name}.faiss".format(index_name=index_name))
        )

        # save docstore and index_to_docstore_id
        with open(path / "{index_name}.pkl".format(index_name=index_name), "wb") as f:
            pickle.dump((self.docstore, self.index_to_docstore_id), f)
   </pre>
