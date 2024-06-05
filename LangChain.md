# LangChain

## Introducción

Cuando conocí *LangChain*, me gustó porque me permitía, con poco código, empezar a usar LLMs locales, cuando yo aún no sabía mucho sobre ese tema. Y luego, al conocerlo un poco mejor, me gustó toda la parte de RAG, ya que tenía implementada la carga de muchos tipos de documentos, e incluso facilitaba "scraping" de varios sitios webes. Pero a día 2024-05-01, y desde hace ya cierto tiempo, me parece difícil de usar para muchos casos de uso, y ya no digamos si quiero entender bien lo que internamente está haciendo, o añadir alguna funcionalidad, ya que su código me resulta tremendamente confuso (no entiendo el por qué de muchas cosas, incluyendo el motivo por el que muchas clases existen, ni la organización de sus directorios; no digo que no tengan sentido (no lo sé), pero no es nada intuitivo para el que no ha estado en su creación desde el principio de todo, e incluso sabiendo bastante sobre LLMs). Por todo eso, quiero ir documentando aquí (poco a poco; WIP, jeje) ese código.

Lo primero de todo, hay que tener en cuenta que ese código (no todo, pero sí la mayor parte, y todo lo más importante a la hora de usarlo) está dividido en 2 grandes grupos, que son los siguientes.
 - [langchain_core](https://github.com/langchain-ai/langchain/tree/master/libs/core/langchain_core)
 - [langchain_community](https://github.com/langchain-ai/langchain/tree/master/libs/community/langchain_community)

Voy a empezar mayormente por el 2º (con algunas excepciones); como he dicho, esta documentación es WIP, así que es posiblel que lo acabe cambiando, pero por ahora quiero empezar por ahí porque esa parte es la más próxima al usuario final, y así por lo menos, al mirar "recetas" (tutoriales de estos cutres (porque no explican nada) y que tanto abundan), con esto será posible entender un poco de eso. Y voy a empezar poniendo un listado de ficheros de ese código (no todos, porque son muchísimos y muchos de ellos sólo son necesarios en ciertos casos de uso muy concretos/específicos), y en el orden en el que me parece más didáctico; en concreto, voy a empezar por los modelos de lenguaje, ya que es el origen, y motivación, de este proyecto, y por los "chats", ya que también es una parte importante de todo esto (ya que la gran popularidad de los LLMs ha sido consecuencia de la aparición de (ese "milagro" llamado) 'ChatGPT'). Por ahora sólo los listo, y poco a poco (a lo largo de varios días o semanas) los voy a ir comentando (o eso espero).
 - [libs/core/langchain_core/language_models/__init__.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/__init__.py)
 - [libs/core/langchain_core/language_models/base.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/base.py)
   La parte principal de este fichero es la clase *BaseLanguageModel*, de la cual a continuación pongo las partes más relevantes. Y además de eso destaco el módulo que usa por defecto para calcular los tokens (toquenizador).
   <pre>
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
   </pre>
 - [libs/core/langchain_core/language_models/llms.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/llms.py)
   Este fichero tiene 1341 líneas a día 2024-06-04, así que, como son muchas, voy a intentar poner sólo lo más relevante y lo más resumido posible.
   Tiene 2 clases:
   - BaseLLM. Cualquier clase descendiente de esta podrá ser usada a través del método 'invoke'. Aquí ese método usa el método _convert_input para adaptar el valor del parámetro 'input', que debe ser "a PromptValue, str, or list of BaseMessages", para ser usado en el método 'generate_prompt' (como un valor de que debe ser instancia de la clase 'PromptValue' o de una descendiente de ella), que a su vez usa/(llama a) el método 'generate', que a su vez usa el método '_generate_helper', que a su vez usa el método '_generate', que a su vez usa el método '_call' (que es un método abstracto que debe ser implementado en cada clase descendiente de esta).
   <pre>
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
   </pre>
   - LLM
   <pre>
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
	</pre>
 - [libs/core/langchain_core/language_models/fake.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/fake.py)
 Como su nombre indica, no es un LLM propiamente dicho sino que simplemente tiene un listado de respuestas predefinidas. Al ser la implementación más simple de una subclase de la clase LLM, es un buen ejemplo de cómo crear subclases de esa clase. Este fichero tiene 2 clases (subclases/descendientes de la clase LLM: FakeListLLM, y FakeStreamingListLLM que es una subclase de 'FakeListLLM'), de las cuales sólo voy a poner aquí la 1ª ('FakeListLLM'; pongo todo su código excepto el método '_acall', porque, para simplificar, en esta documentación estoy ignorando el asincronismo).
 <pre>
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
 </pre>
 - [libs/community/langchain_community/llms/fake.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/fake.py)
   Es igual al anterior (el del paquete "core"), y supongo que lo han copiado desde allí para (en alguna versión) eliminar el anterior.
 - [libs/community/langchain_community/llms/human.py](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/human.py)
   Al igual que el "fake"/falso, este "LLM" tampoco es un LLM propiamente dicho (al menos no "artificial"; este es biológico, jeje) sino que las respuestas son introducidas por el usuario cuando esto es ejecutado. Por tanto, es la 2ª implementación, de una subclase de la clase LLM, más simple. Como es muy corto, y es un buen ejemplo (ni siquiera tiene función asíncrona), aquí pongo todo su código (excepto la importación de módulos).
   <pre>
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
   </pre>
 - [libs/core/langchain_core/language_models/chat_models.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/language_models/chat_models.py)
   Este fichero tiene 2 clases: 'BaseChatModel' y 'SimpleChatModel'.
     - 'BaseChatModel'. La jerarquía de llamadas a funciones es parecida a la de la clase 'BaseModel'. A continuación pongo la jerarquía para esta clase y parte de su código (los métodos principales).
	   + invoke
	     + generate_prompt
	       + generate
		     + _generate_with_cache
			   + _generate (método abstracto)
	 <pre>
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
	 </pre>
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
