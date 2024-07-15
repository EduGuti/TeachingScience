## Introducción

Cuando conocí *LangChain*, me gustó porque me permitía, con poco código, empezar a usar LLMs locales, cuando yo aún no sabía mucho sobre ese tema. Y luego, al conocerlo un poco mejor, me gustó toda la parte de RAG, ya que tenía implementada la carga de muchos tipos de documentos, e incluso facilitaba "scraping" de varios sitios webes. Pero a día 2024-05-01, y desde hace ya cierto tiempo, me parece difícil de usar para muchos casos de uso, y ya no digamos si quiero entender bien lo que internamente está haciendo, o añadir alguna funcionalidad, ya que su código me resulta tremendamente confuso (no entiendo el por qué de muchas cosas, incluyendo el motivo por el que muchas clases existen, ni la organización de sus directorios; no digo que no tengan sentido (no lo sé), pero no es nada intuitivo para el que no ha estado en su creación desde el principio de todo, e incluso sabiendo bastante sobre LLMs). Por todo eso, quiero ir documentando aquí (poco a poco; WIP, jeje) ese código.

Lo primero de todo, hay que tener en cuenta que ese código (no todo, pero sí la mayor parte, y todo lo más importante a la hora de usarlo) está dividido en 2 grandes grupos, que son los siguientes.
 - [langchain_core](https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core)
 - [langchain_community](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community)

Voy a empezar mayormente por el 2º (con algunas excepciones); como he dicho, esta documentación es WIP, así que es posible que lo acabe cambiando, pero por ahora quiero empezar por ahí porque esa parte es la más próxima al usuario final, y así por lo menos, al mirar "recetas" (tutoriales de estos cutres (porque no explican nada) y que tanto abundan), con esto será posible entender un poco de eso. Y voy a empezar poniendo un listado de ficheros de ese código (no todos, porque son muchísimos y muchos de ellos sólo son necesarios en ciertos casos de uso muy concretos/específicos), y en el orden en el que me parece más didáctico. En concreto, voy a empezar por los modelos de lenguaje, ya que es el origen, y motivación, de este proyecto, y por los "chats", ya que también es una parte importante de todo esto (ya que la gran popularidad de los LLMs ha sido consecuencia de la aparición de (ese "milagro" llamado) 'ChatGPT'), y los agrupo en una sección en la que también voy a poner los "prompts", ya que son imprescindibles para usar los LLMs (ya que son las entradas de estos últimos), y las cadenas ("chains"), ya que sirven para unir prompts y LLMs, y es el concepto que da nombre a esta biblioteca ("LangChain").
Empiezo listando los ficheros de código que me parecen más relevantes, y poco a poco los voy comentando y voy poniendo los trozos de código fuente que me parecen más relevantes.

## LLMs, prompts y cadenas

### Ejemplos de uso

#### Con Ollama

En este ejemplo, *db* es un "vectorstore" y es usado para obtener textos, desde una base de datos, para usarlos como contexto de la pregunta (en ese *prompt*). Para más información sobre "vector stores", mirar la última sección de esta página ("Uso de vectores de características"). Y el LLM usado en este ejemplo es *Solar*; el listado completo está en [esta página web](https://ollama.com/library?sort=newest).

```python
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

instruction = "You are a helpful assistant."
prompt_template = f"""
    ### [INST] Instruction: {instruction} Here is context to help:

    {{context}}

    ### QUESTION:
    {{question}} [/INST]
    """
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
llm_ollama = Ollama(model="solar")
llm_chain = LLMChain(llm=llm_ollama, prompt=prompt)
rag_chain = ({"context": db.as_retriever(), "question": RunnablePassthrough()} | llm_chain)
print(rag_chain.invoke("Tell me a joke"))
```

#### Con Groq

Visto en [la documentación de *LangChain*](https://python.langchain.com/docs/integrations/chat/groq).
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

system = "You are a helpful assistant."
human = "{question}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
chain = prompt | chat
print(chain.invoke({"question": "Tell me a joke"}))
```

### Código fuente

 - [libs/core/langchain_core/language_models/__init__.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/__init__.py)
 - [libs/core/langchain_core/language_models/base.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/base.py)
   La parte principal de este fichero es la clase *BaseLanguageModel*, de la cual a continuación pongo las partes más relevantes. Y además de eso destaco el módulo que usa por defecto para calcular los tokens (toquenizador).
   ```python
    @lru_cache(maxsize=None)  # Cache the tokenizer
    def get_tokenizer() -> Any:
        try:
            from transformers import GPT2TokenizerFast  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "Could not import transformers python package. "
                "This is needed in order to calculate get_token_ids. "
                "Please install it with `pip install transformers`."
            )
        # create a GPT-2 tokenizer instance
        return GPT2TokenizerFast.from_pretrained("gpt2")
    ...
    class BaseLanguageModel(
        RunnableSerializable[LanguageModelInput, LanguageModelOutputVar], ABC
    ):
        """Abstract base class for interfacing with language models.
    
        All language model wrappers inherit from BaseLanguageModel.
        """
        ...
        @abstractmethod
        def generate_prompt(
            self,
            prompts: List[PromptValue],
            stop: Optional[List[str]] = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
        ) -> LLMResult:
            """Pass a sequence of prompts to the model and return model generations.
    
            This method should make use of batched calls for models that expose a batched
            API.
    
            Use this method when you want to:
                1. take advantage of batched calls,
                2. need more output from the model than just the top generated value,
                3. are building chains that are agnostic to the underlying language model
                    type (e.g., pure text completion models vs chat models).
    
            Args:
                prompts: List of PromptValues. A PromptValue is an object that can be
                    converted to match the format of any language model (string for pure
                    text generation models and BaseMessages for chat models).
                stop: Stop words to use when generating. Model output is cut off at the
                    first occurrence of any of these substrings.
                callbacks: Callbacks to pass through. Used for executing additional
                    functionality, such as logging or streaming, throughout generation.
                **kwargs: Arbitrary additional keyword arguments. These are usually passed
                    to the model provider API call.
    
            Returns:
                An LLMResult, which contains a list of candidate Generations for each input
                    prompt and additional model provider-specific output.
            """
        ...
        def with_structured_output(
            self, schema: Union[Dict, Type[BaseModel]], **kwargs: Any
        ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
            """Not implemented on this class."""
            # Implement this on child class if there is a way of steering the model to
            # generate responses that match a given schema.
            raise NotImplementedError()
            ...
        def get_token_ids(self, text: str) -> List[int]:
        ...
        def get_num_tokens(self, text: str) -> int:
            return len(self.get_token_ids(text))
        def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
            return sum([self.get_num_tokens(get_buffer_string([m])) for m in messages])
        ...
   ```
 - [libs/core/langchain_core/language_models/llms.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/llms.py)
   Este fichero tiene 1341 líneas a día 2024-06-04, así que, como son muchas, voy a intentar poner sólo lo más relevante y lo más resumido posible.
   Tiene 2 clases:
   - BaseLLM. Cualquier clase descendiente de esta podrá ser usada a través del método 'invoke'. Aquí ese método usa el método _convert_input para adaptar el valor del parámetro 'input', que debe ser "a PromptValue, str, or list of BaseMessages", para ser usado en el método 'generate_prompt' (como un valor de que debe ser instancia de la clase 'PromptValue' o de una descendiente de ella), que a su vez usa/(llama a) el método 'generate', que a su vez usa el método '_generate_helper', que a su vez usa el método '_generate', que a su vez usa el método '_call' (que es un método abstracto que debe ser implementado en cada clase descendiente de esta).
   ```python
   class BaseLLM(BaseLanguageModel[str], ABC):
    """Base LLM abstract interface.

    It should take in a prompt and return a string."""
	
	...
	
	def _convert_input(self, input: LanguageModelInput) -> PromptValue:
        if isinstance(input, PromptValue):
            return input
        elif isinstance(input, str):
            return StringPromptValue(text=input)
        elif isinstance(input, Sequence):
            return ChatPromptValue(messages=convert_to_messages(input))
        else:
            raise ValueError(
                f"Invalid input type {type(input)}. "
                "Must be a PromptValue, str, or list of BaseMessages."
            )

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        config = ensure_config(config)
        return (
            self.generate_prompt(
                [self._convert_input(input)],
                stop=stop,
                callbacks=config.get("callbacks"),
                tags=config.get("tags"),
                metadata=config.get("metadata"),
                run_name=config.get("run_name"),
                run_id=config.pop("run_id", None),
                **kwargs,
            )
            .generations[0][0]
            .text
        )
	
	...
	
	def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Union[Callbacks, List[Callbacks]]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt_strings = [p.to_string() for p in prompts]
        return self.generate(prompt_strings, stop=stop, callbacks=callbacks, **kwargs)
	
	...
	
	def generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Union[Callbacks, List[Callbacks]]] = None,
        *,
        tags: Optional[Union[List[str], List[List[str]]]] = None,
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        run_name: Optional[Union[str, List[str]]] = None,
        run_id: Optional[Union[uuid.UUID, List[Optional[uuid.UUID]]]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Pass a sequence of prompts to a model and return generations.

        This method should make use of batched calls for models that expose a batched
        API.

        Use this method when you want to:
            1. take advantage of batched calls,
            2. need more output from the model than just the top generated value,
            3. are building chains that are agnostic to the underlying language model
                type (e.g., pure text completion models vs chat models).

        Args:
            prompts: List of string prompts.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            callbacks: Callbacks to pass through. Used for executing additional
                functionality, such as logging or streaming, throughout generation.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An LLMResult, which contains a list of candidate Generations for each input
                prompt and additional model provider-specific output.
        """
		
		...
		
        return LLMResult(generations=generations, llm_output=llm_output, run=run_info)
   ```
   - LLM
   ```python
   class LLM(BaseLLM):
    """Simple interface for implementing a custom LLM.

    You should subclass this class and implement the following:

    - `_call` method: Run the LLM on the given prompt and input (used by `invoke`).
    - `_identifying_params` property: Return a dictionary of the identifying parameters
        This is critical for caching and tracing purposes. Identifying parameters
        is a dict that identifies the LLM.
        It should mostly include a `model_name`.

    Optional: Override the following methods to provide more optimizations:

    - `_acall`: Provide a native async version of the `_call` method.
        If not provided, will delegate to the synchronous version using
        `run_in_executor`. (Used by `ainvoke`).
    - `_stream`: Stream the LLM on the given prompt and input.
        `stream` will use `_stream` if provided, otherwise it
        use `_call` and output will arrive in one chunk.
    - `_astream`: Override to provide a native async version of the `_stream` method.
        `astream` will use `_astream` if provided, otherwise it will implement
        a fallback behavior that will use `_stream` if `_stream` is implemented,
        and use `_acall` if `_stream` is not implemented.
    """

    @abstractmethod
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. SHOULD NOT include the prompt.
        """
	
	...
	
	def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        # TODO: add caching here.
        generations = []
        new_arg_supported = inspect.signature(self._call).parameters.get("run_manager")
        for prompt in prompts:
            text = (
                self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
                if new_arg_supported
                else self._call(prompt, stop=stop, **kwargs)
            )
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)
   ```
 - [libs/core/langchain_core/language_models/fake.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/fake.py)
 Como su nombre indica, no es un LLM propiamente dicho sino que simplemente tiene un listado de respuestas predefinidas. Al ser la implementación más simple de una subclase de la clase LLM, es un buen ejemplo de cómo crear subclases de esa clase. Este fichero tiene 2 clases (subclases/descendientes de la clase LLM: FakeListLLM, y FakeStreamingListLLM que es una subclase de 'FakeListLLM'), de las cuales sólo voy a poner aquí la 1ª ('FakeListLLM'; pongo todo su código excepto el método '_acall', porque, para simplificar, en esta documentación estoy ignorando el asincronismo).
 ```python
 class FakeListLLM(LLM):
    """Fake LLM for testing purposes."""

    responses: List[str]
    sleep: Optional[float] = None
    i: int = 0

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake-list"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Return next response"""
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    ...

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"responses": self.responses}
 ```
 - [libs/community/langchain_community/llms/fake.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/fake.py)
   Es igual al anterior (el del paquete "core"), y supongo que lo han copiado desde allí para (en alguna versión) eliminar el anterior.
 - [libs/community/langchain_community/llms/human.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/human.py)
   Al igual que el "fake"/falso, este "LLM" tampoco es un LLM propiamente dicho (al menos no "artificial"; este es biológico, jeje) sino que las respuestas son introducidas por el usuario cuando esto es ejecutado. Por tanto, es la 2ª implementación, de una subclase de la clase LLM, más simple. Como es muy corto, y es un buen ejemplo (ni siquiera tiene función asíncrona), aquí pongo todo su código (excepto la importación de módulos).
```python
def _display_prompt(prompt: str) -> None:
    """Displays the given prompt to the user."""
    print(f"\n{prompt}")  # noqa: T201


def _collect_user_input(
    separator: Optional[str] = None, stop: Optional[List[str]] = None
) -> str:
    """Collects and returns user input as a single string."""
    separator = separator or "\n"
    lines = []

    while True:
        line = input()
        if not line:
            break
        lines.append(line)

        if stop and any(seq in line for seq in stop):
            break
    # Combine all lines into a single string
    multi_line_input = separator.join(lines)
    return multi_line_input


class HumanInputLLM(LLM):
    """User input as the response."""

    input_func: Callable = Field(default_factory=lambda: _collect_user_input)
    prompt_func: Callable[[str], None] = Field(default_factory=lambda: _display_prompt)
    separator: str = "\n"
    input_kwargs: Mapping[str, Any] = {}
    prompt_kwargs: Mapping[str, Any] = {}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Returns an empty dictionary as there are no identifying parameters.
        """
        return {}

    @property
    def _llm_type(self) -> str:
        """Returns the type of LLM."""
        return "human-input"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Displays the prompt to the user and returns their input as a response.

        Args:
            prompt (str): The prompt to be displayed to the user.
            stop (Optional[List[str]]): A list of stop strings.
            run_manager (Optional[CallbackManagerForLLMRun]): Currently not used.

        Returns:
            str: The user's input as a response.
        """
        self.prompt_func(prompt, **self.prompt_kwargs)
        user_input = self.input_func(
            separator=self.separator, stop=stop, **self.input_kwargs
        )

        if stop is not None:
            # I believe this is required since the stop tokens
            # are not enforced by the human themselves
            user_input = enforce_stop_tokens(user_input, stop)
        return user_input
```
 - [libs/core/langchain_core/language_models/chat_models.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/chat_models.py)
   Este fichero tiene 2 clases: 'BaseChatModel' y 'SimpleChatModel'.
     - 'BaseChatModel'. La jerarquía de llamadas a funciones es parecida a la de la clase 'BaseModel'. A continuación pongo la jerarquía para esta clase y parte de su código (los métodos principales).
	   + invoke
	     + generate_prompt
	       + generate
		     + _generate_with_cache
			   + _generate (método abstracto)
```python
class BaseChatModel(BaseLanguageModel[BaseMessage], ABC):
    """Base class for Chat models."""
	 
	...
	 
	def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        config = ensure_config(config)
        return cast(
            ChatGeneration,
            self.generate_prompt(
                [self._convert_input(input)],
                stop=stop,
                callbacks=config.get("callbacks"),
                tags=config.get("tags"),
                metadata=config.get("metadata"),
                run_name=config.get("run_name"),
                run_id=config.pop("run_id", None),
                **kwargs,
            ).generations[0][0],
        ).message
	
	...
	
	def generate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Pass a sequence of prompts to the model and return model generations.

        This method should make use of batched calls for models that expose a batched
        API.

        Use this method when you want to:
            1. take advantage of batched calls,
            2. need more output from the model than just the top generated value,
            3. are building chains that are agnostic to the underlying language model
                type (e.g., pure text completion models vs chat models).

        Args:
            messages: List of list of messages.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            callbacks: Callbacks to pass through. Used for executing additional
                functionality, such as logging or streaming, throughout generation.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An LLMResult, which contains a list of candidate Generations for each input
                prompt and additional model provider-specific output.
        """
	
	...
	
	def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt_messages = [p.to_messages() for p in prompts]
        return self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)
	 </pre>
	 - 'SimpleChatModel'.
	 <pre>
	class SimpleChatModel(BaseChatModel):
    """Simplified implementation for a chat model to inherit from."""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        output_str = self._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @abstractmethod
    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Simpler interface."""
```
 - [libs/core/langchain_core/language_models/fake_chat_models.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/fake_chat_models.py)
 - [libs/community/langchain_community/chat_models/fake.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_models/fake.py)
 - [libs/community/langchain_community/chat_models/human.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_models/human.py)
 - [libs/community/langchain_community/llms/openai.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/openai.py)
 - [libs/community/langchain_community/llms/ollama.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/ollama.py)
 - [libs/community/langchain_community/llms/ctransformers.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/ctransformers.py)
 - [libs/community/langchain_community/llms/gpt4all.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/gpt4all.py)
 - [libs/community/langchain_community/llms/huggingface_hub.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/huggingface_hub.py)
 - [libs/community/langchain_community/llms/huggingface_pipeline.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/huggingface_pipeline.py)
 - [libs/community/langchain_community/llms/huggingface_text_gen_inference.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/huggingface_text_gen_inference.py)
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
 - [libs/langchain/langchain/chains](https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/chains). Las **cadenas** están en este directorio. Son muchas, por lo que aquí sólo voy a ir poniendo algunos ejemplos (por ahora sólo la clase *LLMChain* (del fichero *llm.py*)).
 - [libs/langchain/langchain/chains/llm.py](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/llm.py). <q>Chain that just formats a prompt and calls an LLM.</q>. Tiene la implementación de la clase *LLMChain*, cuya cabecera muestro a continuación.
```python
class LLMChain(Chain):
    """Chain to run queries against LLMs.

    This class is deprecated. See below for an example implementation using
    LangChain runnables:

        .. code-block:: python

            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import PromptTemplate
            from langchain_openai import OpenAI

            prompt_template = "Tell me a {adjective} joke"
            prompt = PromptTemplate(
                input_variables=["adjective"], template=prompt_template
            )
            llm = OpenAI()
            chain = prompt | llm | StrOutputParser()

            chain.invoke("your adjective here")

    Example:
        .. code-block:: python

            from langchain.chains import LLMChain
            from langchain_community.llms import OpenAI
            from langchain_core.prompts import PromptTemplate
            prompt_template = "Tell me a {adjective} joke"
            prompt = PromptTemplate(
                input_variables=["adjective"], template=prompt_template
            )
            llm = LLMChain(llm=OpenAI(), prompt=prompt)
    """
```

## Más utilidades de *LangChain*

 - [libs/core/langchain_core/load/serializable.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/load/serializable.py). Tiene varias clases y la principal es *Serializable(BaseModel, ABC)*.
 - [libs/core/langchain_core/load/mapping.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/load/mapping.py). Sólo tiene un largo diccionario que mapea jerarquía de clases.
 - [libs/core/langchain_core/load/dump.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/load/dump.py). Sólo tiene las funciones *default* (<q>Return a default value for a Serializable object or a SerializedNotImplemented object.</q>), *dumps* (<q>Return a json string representation of an object.</q>) y *dumpd* (<q>Return a json dict representation of an object</q>).
 - [libs/core/langchain_core/load/load.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/load/load.py). Tiene la clase *Reviver* y las funciones *loads* (<q>Revive a LangChain class from a JSON string. Equivalent to `load(json.loads(text))`.</q>) y *load*.
 - [libs/core/langchain_core/messages/base.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/messages/base.py). Entre otras cosas, tiene las clases *BaseMessage(Serializable)* (<q>Messages are the inputs and outputs of ChatModels.</q>) y *BaseMessageChunk(BaseMessage)*.
 - [libs/core/langchain_core/messages/human.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/messages/human.py)
 - [libs/core/langchain_core/messages/chat.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/messages/chat.py). <q>Message that can be assigned an arbitrary speaker (i.e. role).</q>.
 - [libs/core/langchain_core/messages/tool.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/messages/tool.py). <q>Message for passing the result of executing a tool back to a model.</q>.
 - [libs/core/langchain_core/messages/utils.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/messages/utils.py). Tiene 7 funciones para gestionar mensajes.
 - [libs/core/langchain_core/chat_history.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/chat_history.py). Tiene la siguiente cabecera, y las siguientes 2 clases (de las cuales la 2ª la pongo entera por ser corta y un buen ejemplo).
```python
**Chat message history** stores a history of the message interactions in a chat.


**Class hierarchy:**

.. code-block::

    BaseChatMessageHistory --> <name>ChatMessageHistory  # Examples: FileChatMessageHistory, PostgresChatMessageHistory

**Main helpers:**

.. code-block::

    AIMessage, HumanMessage, BaseMessage

...

class BaseChatMessageHistory(ABC):
    """Abstract base class for storing chat message history.

    Implementations guidelines:

    Implementations are expected to over-ride all or some of the following methods:

    * add_messages: sync variant for bulk addition of messages
    * aadd_messages: async variant for bulk addition of messages
    * messages: sync variant for getting messages
    * aget_messages: async variant for getting messages
    * clear: sync variant for clearing messages
    * aclear: async variant for clearing messages

    add_messages contains a default implementation that calls add_message
    for each message in the sequence. This is provided for backwards compatibility
    with existing implementations which only had add_message.

    Async variants all have default implementations that call the sync variants.
    Implementers can choose to over-ride the async implementations to provide
    truly async implementations.

    Usage guidelines:

    When used for updating history, users should favor usage of `add_messages`
    over `add_message` or other variants like `add_user_message` and `add_ai_message`
    to avoid unnecessary round-trips to the underlying persistence layer.

    Example: Shows a default implementation.

        .. code-block:: python

            class FileChatMessageHistory(BaseChatMessageHistory):
                storage_path:  str
                session_id: str

               @property
               def messages(self):
                   with open(os.path.join(storage_path, session_id), 'r:utf-8') as f:
                       messages = json.loads(f.read())
                    return messages_from_dict(messages)

               def add_messages(self, messages: Sequence[BaseMessage]) -> None:
                   all_messages = list(self.messages) # Existing messages
                   all_messages.extend(messages) # Add new messages

                   serialized = [message_to_dict(message) for message in all_messages]
                   # Can be further optimized by only writing new messages
                   # using append mode.
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       json.dump(f, messages)

               def clear(self):
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       f.write("[]")
    """

    messages: List[BaseMessage]
    """A property or attribute that returns a list of messages.

    In general, getting the messages may involve IO to the underlying
    persistence layer, so this operation is expected to incur some
    latency.
    """

...

class InMemoryChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history.

    Stores messages in an in memory list.
    """

    messages: List[BaseMessage] = Field(default_factory=list)

    async def aget_messages(self) -> List[BaseMessage]:
        return self.messages

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        self.messages.append(message)

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add messages to the store"""
        self.add_messages(messages)

    def clear(self) -> None:
        self.messages = []

    async def aclear(self) -> None:
        self.clear()
```
 - [libs/community/langchain_community/chat_message_histories/file.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_message_histories/file.py)
 - [libs/community/langchain_community/chat_message_histories/mongodb.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_message_histories/mongodb.py)
 - [libs/community/langchain_community/chat_message_histories/postgres.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_message_histories/postgres.py)
 - [libs/community/langchain_community/chat_message_histories/neo4j.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_message_histories/neo4j.py)
 - [libs/community/langchain_community/chat_message_histories/redis.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_message_histories/redis.py)
 - [libs/community/langchain_community/chat_message_histories/sql.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_message_histories/sql.py)
 - [libs/core/langchain_core/chat_sessions.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/chat_sessions.py). <q>**Chat Sessions** are a collection of messages and function calls.</q>. Sólo tiene la siguiente clase.
```python
class ChatSession(TypedDict, total=False):
    """Chat Session represents a single
    conversation, channel, or other group of messages."""

    messages: Sequence[BaseMessage]
    """The LangChain chat messages loaded from the source."""
    functions: Sequence[dict]
    """The function calling specs for the messages."""
```
 - [libs/core/langchain_core/chat_loaders.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/chat_loaders.py). Sólo tiene la siguiente clase.
```python
class BaseChatLoader(ABC):
    """Base class for chat loaders."""

    @abstractmethod
    def lazy_load(self) -> Iterator[ChatSession]:
        """Lazy load the chat sessions."""

    def load(self) -> List[ChatSession]:
        """Eagerly load the chat sessions into memory."""
        return list(self.lazy_load())
```
 - [libs/community/langchain_community/chat_loaders/utils.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_loaders/utils.py)
 - [libs/community/langchain_community/chat_loaders/gmail.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_loaders/gmail.py)
 - [libs/community/langchain_community/chat_loaders/telegram.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_loaders/telegram.py)
 - [libs/community/langchain_community/chat_loaders/whatsapp.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/chat_loaders/whatsapp.py)
 - [libs/community/langchain_community/cache.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/cache.py)
 - [core/langchain_core/documents/base.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/documents/base.py). Sólo tiene la clase *Document(Serializable)* (<q>Class for storing a piece of text and associated metadata.</q>).
 - [libs/core/langchain_core/documents/compressor.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/documents/compressor.py). Sólo tiene la clase *BaseDocumentCompressor(BaseModel, ABC)* (<q>Base class for document compressors.</q>).
 - [libs/community/langchain_community/utilities](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/utilities). A día 2024-06-11, este directorio tiene **74 ficheros** (incluyendo el *__init__.py*). Contiene clases que son usadas en las clases ("utilidades") definidas en los directorios *libs/community/langchain_community/tools*, ¿*libs/community/langchain_community/agent_toolkits*? (ToDo: Comprobar esto.), *libs/community/langchain_community/document_loaders* y *libs/community/langchain_community/retrievers*; por tanto, sería interesante hacer grupos de temas/servicios, juntando/agrupando esos 3||4 tipos de clases (en principio, voy a ir creando algunos, y colocándolos tras estas 4 secciones). A groso modo, la diferencia entre esos 4 tipos es la siguiente: Las herramientas ("tools") son creadas para ser usadas por agentes de *LangChain*, los *toolkits* son conjuntos de herramientas (cada uno tiene una clase con un método que devuelve un listado (de *Python*) de objetos de la clase *BaseTool* (o clase descenciente de esa, claro)), los *retrievers* están pensados para obtener textos (a través de la clase *Document*) desde cualquier tipo de fuente (en su código fuente está comentado que los *vector stores* pueden ser considerados un tipo de *retrievers*, pero están en un directorio a parte), y los *document loaders* están pensados para obtener el texto (a través de la clase *Document*, pero en este caso son 2 nombres/términos iguales pero que hacen referencia a conceptos diferentes (ejemplo de polisemia, jeje)) a partir de esas piezas de software que en informática solemos llamar documentos (a día 2024-06-11 no sé cómo definir eso bien, a pesar de que llevo décadas usando ese término con ese significado y que lo veo usar así en todo lo que conozco de informática (es decir, de manera intuitiva sé lo que es, al igual que cualquier persona que sepa algo de informática (con que sólo sepa ofimática, es suficiente), pero sorprendentemente no se me ocurre cómo definirlo bien/(sin ambigüedades y con/(partiendo de)/(usando sólo) terminología más básica)).
 - [libs/community/langchain_community/utilities/stackexchange.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/stackexchange.py). Sólo contiene la clase *StackExchangeAPIWrapper(BaseModel)*, que usa el paquete *stackapi* y su principal método es *run* (devuelve una cadena de texto) (no tiene método *load*).
 - [libs/community/langchain_community/utilities/sql_database.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/sql_database.py). Su única clase es *SQLDatabase*, que usa la biblioteca *sqlalchemy* y su principal método es *run* (devuelve una cadena de texto) (no tiene método *load*).
 - [libs/community/langchain_community/utilities/requests.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/requests.py). Contiene las clases *Requests(BaseModel)* y *GenericRequestsWrapper(BaseModel)*, *JsonRequestsWrapper(GenericRequestsWrapper)* y *TextRequestsWrapper(GenericRequestsWrapper)*. Ninguna de ellas tiene métodos ni *run* ni *load*, sino métodos (de *Python*, y excepto las 2 últimas) para usar métodos de HTTP.
 - [libs/community/langchain_community/utilities/reddit_search.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/reddit_search.py). Su única clase es *RedditSearchAPIWrapper(BaseModel)*, que usa la biblioteca *praw* y sus 2 principal métodos son *run* (devuelve una cadena de texto) y *results* (devuelve un listado de diccionarios de *Python*). **Usa clave ("key") de ese servicio web** (no sé si es gratuito o no).
 - [libs/community/langchain_community/utilities/python.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/python.py). Su única clase es *PythonREPL(BaseModel)*, cuyo principal método son *run* (devuelve una cadena de texto) (no tiene método *load*). Ejecuta comandos usando la función (de *Python*) *exec*, y también tiene una gestión de colas para eso.
 - [libs/community/langchain_community/utilities/merriam_webster.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/merriam_webster.py). Su única clase es *MerriamWebsterAPIWrapper(BaseModel)*, cuyo principal método es *run* (devuelve una cadena de texto) (no tiene método) *load*). Obtiene la información a través de URLs. **Usa clave ("key") de ese servicio web** (no sé si es gratuito o no).
 - [libs/community/langchain_community/utilities/bibtex.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/bibtex.py). Su única clase es *BibtexparserWrapper(BaseModel)*, que usa la biblioteca *bibtexparser* y sus 2 principales métodos son *load_bibtex_entries(self, path: str) -> List[Dict[str, Any]]* y *get_metadata(self, entry: Mapping[str, Any], load_extra: bool = False) -> Dict[str, Any]*.
 - [libs/community/langchain_community/utilities/github.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/github.py). Su única clase es *GitHubAPIWrapper(BaseModel)*, que usa la biblioteca *github*. No tiene ni método *run*, ni método *load*, sino varios para hacer gestiones de repositorios (y, por tanto, **hace uso de clave/key de usuario**; obviamente, en este casi sí que sé que es gratuito ;-) ).
 - [libs/community/langchain_community/utilities/openweathermap.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/openweathermap.py). Su única clase es *OpenWeatherMapAPIWrapper(BaseModel)*, que usa la biblioteca *pyowm* y su principal método es *run* (devuelve una cadena de texto) (no tiene método *load*). **Usa clave ("key") de ese servicio web** (no sé si es gratuito o no).
 - [libs/core/langchain_core/tools.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/tools.py). Tiene la siguiente cabecera, y entre otras cosas tiene las clases *BaseTool(RunnableSerializable[Union[str, Dict], Any])* (jerarquía de métodos (sólo los principales): *invoke*->*run*->*_run* y este último debe estar implementado en cada uno de sus hijos), *Tool(BaseTool)*, * StructuredTool(BaseTool)* y *BaseToolkit(BaseModel, ABC)*.
```python
**Tools** are classes that an Agent uses to interact with the world.

Each tool has a **description**. Agent uses the description to choose the right
tool for the job.

**Class hierarchy:**

.. code-block::

    RunnableSerializable --> BaseTool --> <name>Tool  # Examples: AIPluginTool, BaseGraphQLTool
                                          <name>      # Examples: BraveSearch, HumanInputRun

**Main helpers:**

.. code-block::

    CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
```
 - [libs/community/langchain_community/tools/sql_database/prompt.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/sql_database/prompt.py). Sólo tiene la plantilla que será usada para crear la descripción que el agente, a través de un LLM, leerá para saber usar la herramienta.
 - [libs/community/langchain_community/tools/sql_database/tool.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/sql_database/tool.py). Usa la clase *SQLDatabase*, del correspondiente fichero de utilidades de *LangChain*. Tiene las clases *BaseSQLDatabaseTool(BaseModel)* (<q>Base tool for interacting with a SQL database.</q>), *QuerySQLDataBaseTool(BaseSQLDatabaseTool, BaseTool)* (<q>Tool for querying a SQL database.</q>), *InfoSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool)* (<q>Tool for getting metadata about a SQL database.</q>), *ListSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool)* (<q>Tool for getting tables names.</q>), ** (<q></q>) y *QuerySQLCheckerTool(BaseSQLDatabaseTool, BaseTool)* (<q>Use an LLM to check if a query is correct. Adapted from https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/</q>); más 4 secundarias/(de apoyo) y muy pequeñas.
 - [libs/community/langchain_community/tools/human/tool.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/human/tool.py). Es una herramienta muy simple que simplemente hace una pregunta a un humano y devuelve la respuesta que el humano le escribe. La clase ahí implementada es *HumanInputRun(BaseTool)*.
 - [libs/community/langchain_community/tools/json/tool.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/json/tool.py). Tiene 3 clases: *JsonSpec(BaseModel)* (<q>Base class for JSON spec.</q>; esta no es una "herramienta" de *LangChain*), *JsonListKeysTool(BaseTool)* (<q>Tool for listing keys in a JSON spec.</q>) y *JsonGetValueTool(BaseTool)* (<q>Tool for getting a value in a JSON spec.</q>).
 - [libs/community/langchain_community/tools/requests/tool.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/requests/tool.py). Usa la clase *GenericRequestsWrapper*, que es una "utilidad" de *LangChain*. Tiene 6 clases: *BaseRequestsTool(BaseModel):* (<q>Base class for requests tools.</q>; no es una "herramienta" sino una clase usada como madre de las otras 5, que sí lo son, y sólo sirve para mostrar un aviso de los peligros de seguridad de usar alguna, o varias, de esas 5 herramientas), *RequestsGetTool(BaseRequestsTool, BaseTool)* (<q>Tool for making a GET request to an API endpoint.</q>), *RequestsPostTool(BaseRequestsTool, BaseTool)* (<q>Tool for making a POST request to an API endpoint.</q>), * RequestsPatchTool(BaseRequestsTool, BaseTool)* (<q>Tool for making a PATCH request to an API endpoint.</q>), *RequestsPutTool(BaseRequestsTool, BaseTool)* (<q>Tool for making a PUT request to an API endpoint.</q>) y *RequestsDeleteTool(BaseRequestsTool, BaseTool)* (<q>Tool for making a DELETE request to an API endpoint.</q>).
 - [libs/community/langchain_community/tools/playwright](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/tools/playwright) (en este directorio a día 2024-05-01 hay 10 ficheros)
 - [libs/community/langchain_community/tools/playwright/base.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/playwright/base.py). Usa la biblioteca/paquete *playwright*. Sólo tiene 1 clase: *BaseBrowserTool(BaseTool)* (<q>Base class for browser tools.</q>), que no es una "herramienta" sino una clase madre de las otras y que sólo es usada para gestionar la importación (desde ese paquete) del navegador.
 - [libs/community/langchain_community/tools/playwright/utils.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/playwright/utils.py). <q>Utilities for the Playwright browser tools.</q>. Tiene las siguientes funciones: *aget_current_page*, *get_current_page* (usa el método *new_context* de la clase *Browser*, y luego *context.new_page()* (ni no hay ninguno; si hay varios, usa el 1º), a menos que ya haya alguna página en el navegador, y devuelve la página (la última, en el caso de que ya haya más de 1 abierta en el navegador)), *create_async_playwright_browser*, *create_sync_playwright_browser* (<q>Create a playwright browser</q>. Sólo tiene 3 sentencias: <code>from playwright.sync_api import sync_playwright</code>, <code>browser = sync_playwright().start()</code> y <code>browser.chromium.launch(headless=headless, args=args)</code>) y *run_async*.
 - [libs/community/langchain_community/tools/playwright/click.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/playwright/click.py). La clase principal (la "herramienta") es *ClickTool(BaseBrowserTool)* (<q>Tool for clicking on an element with the given CSS selector.</q>). Usa el método *click* de la clase *Page* de *PlayWright*.
 - [libs/community/langchain_community/tools/playwright/current_page.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/playwright/current_page.py). La clase principal (la "herramienta") es *CurrentWebPageTool(BaseBrowserTool)* (<q>Tool for getting the URL of the current webpage.</q>; simplemente usa la función 'get_current_page' (o su equivalente asíncrona), del fichero 'utils.py' de este mismo directorio, y devuelve el URL de la página que está activa en el navegador).
 - [libs/community/langchain_community/tools/playwright/extract_hyperlinks.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/playwright/extract_hyperlinks.py). La clase principal (la "herramienta") es *ExtractHyperlinksTool(BaseBrowserTool)* (<q>Extract all hyperlinks on the page.</q>). Usa el paquete *beautifulsoup4* (*BeautifulSoup*; módulo *bs4*) para "Find all the anchor elements and extract their href attributes", y "Return the list of links as a JSON string". Por tanto, realmente no encuentra/devuelve todos los enlaces de la página (la actualmente activa en el navegador), ya que hay enlaces que (por desgracia, ya que según el estándar debería ser así) no usan esos elementos de HTML; pero al menos extrae (para los de esos elementos) tanto los URL absolutos como los relativos.
 - [libs/community/langchain_community/tools/playwright/extract_text.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/playwright/extract_text.py). La clase principal (la "herramienta") es *ExtractTextTool(BaseBrowserTool)* (<q>Tool for extracting all the text on the current webpage.</q>). También usa el paquete *beautifulsoup4*; en concreto, en este caso usa su método *stripped_strings*, y devuelve la cadena de texto formada por todos esos elementos unidos con un espacio (entre cada uno de ellos). Para obtener el contenido de la página, usa el método *content* de la clase *Page* de *PlayWright*.
 - [libs/community/langchain_community/tools/playwright/get_elements.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/playwright/get_elements.py). La clase principal (la "herramienta") es *GetElementsTool(BaseBrowserTool)* (<q>Tool for getting elements in the current web page matching a CSS selector.</q>). Para obtener los elementos, usa el método *query_selector_all* de la clase *Page* de *PlayWright*.
 - [libs/community/langchain_community/tools/playwright/navigate.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/playwright/navigate.py). La clase principal (la "herramienta") es *NavigateTool(BaseBrowserTool)* (<q>Tool for navigating a browser to a URL.</q>). Usa el método *goto* de la clase *Page* de *PlayWright*.
 - [libs/community/langchain_community/tools/playwright/navigate_back.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/playwright/navigate_back.py). La clase principal (la "herramienta") es *NavigateBackTool(BaseBrowserTool)* (<q>Navigate back to the previous page in the browser history.</q>). Usa el método *go_back* de la clase *Page* de *PlayWright*.
 - [libs/community/langchain_community/agent_toolkits/sql/base.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/agent_toolkits/sql/base.py)
 - [libs/community/langchain_community/agent_toolkits/sql/prompt.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/agent_toolkits/sql/prompt.py)
 - [libs/community/langchain_community/agent_toolkits/sql/toolkit.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/agent_toolkits/sql/toolkit.py)
 - [libs/community/langchain_community/agent_toolkits/playwright/toolkit.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/agent_toolkits/playwright/toolkit.py)
 - [libs/core/langchain_core/document_loaders/base.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/document_loaders/base.py). Tiene 2 clases: *BaseLoader(ABC)* (<q>Implementations should implement the lazy-loading method using generators to avoid loading all Documents into memory at once. `load` is provided just for user convenience and should not be overridden.</q>) y *BaseBlobParser(ABC)* (<q>A blob parser provides a way to parse raw data stored in a blob into one or more documents. The parser can be composed with blob loaders, making it easy to reuse a parser independent of how the blob was originally loaded.</q>).
 - [libs/core/langchain_core/document_loaders/blob_loaders.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/document_loaders/blob_loaders.py). Tiene 2 clases: *Blob(BaseModel)* (<q>Blob represents raw data by either reference or value. Provides an interface to materialize the blob in different representations, and help to decouple the development of data loaders from the downstream parsing of the raw data. Inspired by: https://developer.mozilla.org/en-US/docs/Web/API/Blob</q>) y *BlobLoader(ABC)* (<q>Abstract interface for blob loaders implementation. Implementer should be able to load raw content from a storage system according to some criteria and return the raw content lazily as a stream of blobs.</q>).
 - [libs/community/langchain_community/document_loaders/blob_loaders](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/document_loaders/blob_loaders) (4 ficheros)
 - [libs/community/langchain_community/document_loaders/parsers](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/document_loaders/parsers) (11 ficheros y 2 subdirectorios)
 - [libs/community/langchain_community/document_loaders/parsers/html/bs4.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_loaders/parsers/html/bs4.py)
 - [libs/community/langchain_community/document_loaders/sql_database.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_loaders/sql_database.py)
 - [libs/community/langchain_community/document_loaders/web_base.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_loaders/web_base.py)
 - [libs/community/langchain_community/document_loaders/sql_database.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_loaders/sql_database.py)
 - [libs/core/langchain_core/documents/transformers.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/documents/transformers.py). Sólo tiene la clase *BaseDocumentTransformer(ABC)* (<q>Abstract base class for document transformation systems.</q>. <q>A document transformation system takes a sequence of Documents and returns a sequence of transformed Documents.</q>). Esa clase sólo tiene 2 métodos (*transform_documents* y su versión asíncrona).
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
 - [libs/core/langchain_core/retrievers.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/retrievers.py). Tiene la siguiente cabecera y la clase *BaseRetriever* (de la cual aquí pongo su cabecera).
```python
**Retriever** class returns Documents given a text **query**.

It is more general than a vector store. A retriever does not need to be able to
store documents, only to return (or retrieve) it. Vector stores can be used as
the backbone of a retriever, but there are other types of retrievers as well.

**Class hierarchy:**

.. code-block::

    BaseRetriever --> <name>Retriever  # Examples: ArxivRetriever, MergerRetriever

**Main helpers:**

.. code-block::

    RetrieverInput, RetrieverOutput, RetrieverLike, RetrieverOutputLike,
    Document, Serializable, Callbacks,
    CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
```
```python
class BaseRetriever(RunnableSerializable[RetrieverInput, RetrieverOutput], ABC):
    """Abstract base class for a Document retrieval system.


    A retrieval system is defined as something that can take string queries and return
    the most 'relevant' Documents from some source.

    Usage:

    A retriever follows the standard Runnable interface, and should be used
    via the standard runnable methods of `invoke`, `ainvoke`, `batch`, `abatch`.

    Implementation:

    When implementing a custom retriever, the class should implement
    the `_get_relevant_documents` method to define the logic for retrieving documents.

    Optionally, an async native implementations can be provided by overriding the
    `_aget_relevant_documents` method.

    Example: A retriever that returns the first 5 documents from a list of documents

        .. code-block:: python

            from langchain_core import Document, BaseRetriever
            from typing import List

            class SimpleRetriever(BaseRetriever):
                docs: List[Document]
                k: int = 5

                def _get_relevant_documents(self, query: str) -> List[Document]:
                    \"\"\"Return the first k documents from the list of documents\"\"\"
                    return self.docs[:self.k]

                async def _aget_relevant_documents(self, query: str) -> List[Document]:
                    \"\"\"(Optional) async native implementation.\"\"\"
                    return self.docs[:self.k]

    Example: A simple retriever based on a scitkit learn vectorizer

        .. code-block:: python

            from sklearn.metrics.pairwise import cosine_similarity

            class TFIDFRetriever(BaseRetriever, BaseModel):
                vectorizer: Any
                docs: List[Document]
                tfidf_array: Any
                k: int = 4

                class Config:
                    arbitrary_types_allowed = True

                def _get_relevant_documents(self, query: str) -> List[Document]:
                    # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
                    query_vec = self.vectorizer.transform([query])
                    # Op -- (n_docs,1) -- Cosine Sim with each doc
                    results = cosine_similarity(self.tfidf_array, query_vec).reshape((-1,))
                    return [self.docs[i] for i in results.argsort()[-self.k :][::-1]]
    """  # noqa: E501
```
 - [libs/community/langchain_community/retrievers/knn.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/retrievers/knn.py)
 - [libs/community/langchain_community/retrievers/svm.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/retrievers/svm.py)
 - [libs/community/langchain_community/retrievers/llama_index.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/retrievers/llama_index.py)
 - Grupos de utilidades, herramientas, cargadores de documentos y "retrievers" (ToDo: Crear una sección para esto en este documento):
 	- [libs/community/langchain_community/utilities/duckduckgo_search.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/duckduckgo_search.py). Sólo tiene la clase *DuckDuckGoSearchAPIWrapper(BaseModel)*, la cual usa el paquete *duckduckgo_search* y sus 2 principales métodos son *run* (devuelve una cadena de texto) y *results* (devuelve un listado de diccionarios de *Python*).
 	- [libs/community/langchain_community/tools/ddg_search/tool.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/ddg_search/tool.py)
 	- [libs/community/langchain_community/utilities/wikipedia.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/wikipedia.py). Sólo contiene la clase *WikipediaAPIWrapper(BaseModel)*, que usa el paquete *wikipedia* y sus 2 principales métodos son *run* (devuelve una cadena de texto) y *load* (devuelve un listado de documentos).
 	- [libs/community/langchain_community/tools/wikipedia/tool.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/wikipedia/tool.py)
 	- [libs/community/langchain_community/document_loaders/wikipedia.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_loaders/wikipedia.py)
 	- [libs/community/langchain_community/retrievers/wikipedia.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/retrievers/wikipedia.py)
 	- [libs/community/langchain_community/utilities/wikidata.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/wikidata.py). Sólo tiene la clase *WikidataAPIWrapper(BaseModel)*, que usa el paquete *mediawikiapi* y sus 2 principales métodos son *run* (devuelve una cadena de texto) y *load* (devuelve un listado de documentos).
 	- [libs/community/langchain_community/tools/wikidata/tool.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/tools/wikidata/tool.py)
 	- [libs/community/langchain_community/utilities/arxiv.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/arxiv.py). Sólo tiene la clase *ArxivAPIWrapper(BaseModel)*, la cual usa el paquete *arxiv* y sus 2 principales métodos son *run* (devuelve una cadena de texto) y *load* (devuelve un listado de documentos). A través del método *lazy_load* (usado por el método *load*), usa el paquete *pymupdf* para extraer todo el texto de cada "paper" bajado; en el texto generado&devuelto por *run* sólo va fecha, título, autores y resumen, de cada "paper".
 	- [libs/community/langchain_community/retrievers/arxiv.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/retrievers/arxiv.py)
 	- [libs/community/langchain_community/utilities/pubmed.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/utilities/pubmed.py) Sólo contiene la clase *PubMedAPIWrapper(BaseModel)*, que usa el paquete *xmltodict* (para extraer información del documento web (¿es una página web?) obtenido directamente a través de un URL (por "paper") creado por esta clase, y no sé si es posible obtener de esta forma el texto completo del correspondiente "paper", pero esta clase sólo extrae metadatos del "paper") y sus 2 principales métodos son *run* (devuelve una cadena de texto) y *load* (devuelve un listado de documentos).
 	- [libs/community/langchain_community/retrievers/pubmed.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/retrievers/pubmed.py)
 - [libs/core/langchain_core/runnables](https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/runnables) (a día 2024-06-08 tiene 17 ficheros)
 - [libs/core/langchain_core/prompts](https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core/prompts) (a día 2024-06-08 tiene 11 ficheros)
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
 - [libs/core/langchain_core/agents.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/agents.py). Contiene las clases *AgentAction*, *AgentActionMessageLog*, *AgentStep* y *AgentFinish*, y las funciones *_convert_agent_action_to_messages*, *_convert_agent_observation_to_messages* y *_create_function_message*. Su cabecera es la siguiente.
```python
**Agent** is a class that uses an LLM to choose a sequence of actions to take.

In Chains, a sequence of actions is hardcoded. In Agents,
a language model is used as a reasoning engine to determine which actions
to take and in which order.

Agents select and use **Tools** and **Toolkits** for actions.

**Class hierarchy:**

.. code-block::

    BaseSingleActionAgent --> LLMSingleActionAgent
                              OpenAIFunctionsAgent
                              XMLAgent
                              Agent --> <name>Agent  # Examples: ZeroShotAgent, ChatAgent


    BaseMultiActionAgent  --> OpenAIMultiFunctionsAgent


**Main helpers:**

.. code-block::

    AgentType, AgentExecutor, AgentOutputParser, AgentExecutorIterator,
    AgentAction, AgentFinish, AgentStep
```

## Uso de vectores de características

A día 2024-02-01, del [código fuente de la gestión, de sistemas de almacenamiento de vectores, de *LangChain*](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community/vectorstores), he mirado el contenido de los siguientes ficheros (de los 76 que hay ahí, si no he contado mal), los cuales a continuación listo en orden desde el más simple hasta el más complejo (en igualdad de complejidad, al menos en cuanto a cantidad de funcionalidades y de manera aproximada, pongo delante los que usan extensiones de otras bases de datos más generales y populares, como por ejemplo *SQLite*), y luego iré creando subsecciones para comentarlos.
 - [sklearn.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/sklearn.py)
 - [faiss.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/faiss.py)
 - [sqlitevss.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/sqlitevss.py)
 - [pgvector.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/pgvector.py)
 - [chroma.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/chroma.py)
 - [qdrant.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/qdrant.py)

Son subclases de la clase [VectorStore](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/vectorstores/base.py#L70).

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

```python
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
```

A continuación pongo un esquema (no completo, sino sólo de los métodos que me parecen más relevantes) con las dependencias de esas 4 funciones usadas para crear ese almacenamiento de vectores.
Aquí no pongo la descripción, de cada método/clase y que está en el código fuente, para que este esquema ocupe menos y porque es posible verlo siguiendo el correspondiente enlace.
AVISO: Los enlaces indican la línea en la que el correspondiente métido está definido, en el código fuente de *LangChain*, a día 2024-01-30, pero puede ser que cambien eso y yo no voy a estar muy pendiente de actualizar esto.
WIP:
 - **[from_texts](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/faiss.py#L933)**
   ```python
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
   ```
   - [__from](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/faiss.py#L872)
     ```python
     @classmethod
     def __from(
         cls,
         texts: Iterable[str],
         embeddings: List[List[float]],
         embedding: Embeddings,
         metadatas: Optional[Iterable[dict]] = None,
         ids: Optional[List[str]] = None,
         normalize_L2: bool = False,
         distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
         **kwargs: Any,
     ) -> FAISS:
         faiss = dependable_faiss_import()
         if distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
             index = faiss.IndexFlatIP(len(embeddings[0]))
         else:
             # Default to L2, currently other metric types not initialized.
             index = faiss.IndexFlatL2(len(embeddings[0]))
         docstore = kwargs.pop("docstore", InMemoryDocstore())
         index_to_docstore_id = kwargs.pop("index_to_docstore_id", {})
         vecstore = cls(
             embedding,
             index,
             docstore,
             index_to_docstore_id,
             normalize_L2=normalize_L2,
             distance_strategy=distance_strategy,
             **kwargs,
         )
         vecstore.__add(texts, embeddings, metadatas=metadatas, ids=ids)
         return vecstore
     ```
     - [dependable_faiss_import](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/faiss.py#L38) simplemente <q>Import faiss if available, otherwise raise error. If FAISS_NO_AVX2 environment variable is set, it will be considered to load FAISS with no AVX2 optimization.</q>.
     - [__add](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/faiss.py#L168)
       ```python
       def __add(
           self,
           texts: Iterable[str],
           embeddings: Iterable[List[float]],
           metadatas: Optional[Iterable[dict]] = None,
           ids: Optional[List[str]] = None,
       ) -> List[str]:
           faiss = dependable_faiss_import()

           if not isinstance(self.docstore, AddableMixin):
               raise ValueError(
                   "If trying to add texts, the underlying docstore should support "
                   f"adding items, which {self.docstore} does not"
               )

           _len_check_if_sized(texts, metadatas, "texts", "metadatas")
           _metadatas = metadatas or ({} for _ in texts)
           documents = [
               Document(page_content=t, metadata=m) for t, m in zip(texts, _metadatas)
           ]

           _len_check_if_sized(documents, embeddings, "documents", "embeddings")
           _len_check_if_sized(documents, ids, "documents", "ids")

           if ids and len(ids) != len(set(ids)):
               raise ValueError("Duplicate ids found in the ids list.")

           # Add to the index.
           vector = np.array(embeddings, dtype=np.float32)
           if self._normalize_L2:
               faiss.normalize_L2(vector)
           self.index.add(vector)

           # Add information to docstore and index.
           ids = ids or [str(uuid.uuid4()) for _ in texts]
           self.docstore.add({id_: doc for id_, doc in zip(ids, documents)})
           starting_len = len(self.index_to_docstore_id)
           index_to_id = {starting_len + j: id_ for j, id_ in enumerate(ids)}
           self.index_to_docstore_id.update(index_to_id)
           return ids
       ```
 - **[load_local](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/faiss.py#L1085)**
   ```python
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
   ```
 - **[add_texts](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/faiss.py#L207)**
   ```python
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
   ```
 - **[save_local](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/vectorstores/faiss.py#L1063)**
   ```python
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
   ```
