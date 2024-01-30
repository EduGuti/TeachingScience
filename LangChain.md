# LangChain

## Uso de vectores de características

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
