# Introducción

Por ahora sólo voy a usar este fichero para ir compartiendo notas cortas sobre este tema. Espero que más adelante yo pueda escribir algo bien redactado sobre este tema.

## Introducción al "prompting".
En el contexto de LLMs (modelos de lenguaje grandes, o modelos grandes de lenguaje), un "prompt" es el texto que el correspondiente modelo recibe para generar una respuesta. A groso modo, puede ser pensado como el texto que 
el usuario escribe y envía, por ejemplo en ChatGPT, pero en realidad la aplicación suele añadir más texto, que no es mostrado al usuario. Este tema ha cogido tanta popularidad, que actualmente (año 2023) está de moda la 
llamada "ingeniería de prompting".

Como **introducción a este tema**, recomiendo el **vídeo** [Tutorial de ingeniería rápida: respuestas maestras de ChatGPT y LLM](https://www.youtube.com/watch?v=_ZvnD73m40o).

**Ejemplo** creado por mí el día 2023-10-16 **con *ChatGPT Plus*** (en España, con GPT-4 y seleccionada la búsqueda con Bing (que a ese día aún está como herramienta "beta"); actualmente, *ChatGPT Plus* cuesta 20 $ por mes sin impuestos y 24,40 $ por mes con impuestos, lo cual vienen a ser entre 22 y 23 €/mes):
**Ejemplo de estudio de clasificación biológica**: "[Eres un biólogo especializado en clasificación de seres vivos. Conoces las características de cada rango jerárquico necesario para clasificar cada especie, desde el superior hasta la especie o subespecie. Estás familiarizado con diferentes tipos de clasificación, pero puedes consultar Internet para ampliar o actualizar tus conocimientos. Muestra un listado de todos los rangos de la siguiente especie, poniendo junto a cada rango (nivel jerárquico) las características de ese rango y sólo de ese rango, es decir, sin repetirlas en varios/otros rangos o cualquier otra descripción. Intenta ser lo más específico posible; es decir, pon la mayor cantidad de características posible, y para eso elige el tipo de clasificación más completo que conozcas, y al comienzo de tu respuesta indica cuál has elegido. Como tienes que buscar mucho en Internet. Muestra 2 rangos de cada vez. ESPECIE: Plexippus paykulli](https://chat.openai.com/share/b04b695a-c010-499a-8dc6-3d802d64f9ee)". Podeis comparar ese resultado con [el correspondiente artículo/entrada en *Wikipedia* en inglés](https://en.wikipedia.org/wiki/Plexippus_paykulli); tenez en cuenta que yo lo estoy mirando a día 2023-10-16 y su contenido puede cambiar (en [su página de historial de cambios](https://en.wikipedia.org/w/index.php?title=Plexippus_paykulli&action=history) podeis ver si ha cambiado). ¿Cuál os gusta más?

Y para hacer eso **con imágenes** (característica/funcionalidad nueva en ChatGPT Plus; desde el día 2023-09-23 en EEUU y desde el día 2023-10-14 (al menos yo la vi ese día) en España, y supongo que en toda Europa), hay un estupendo paper/artículo con **ejemplos de** creación de **prompts que contienen texto e imágenes**:
 - [Página del **paper** en Arxiv](https://arxiv.org/abs/2309.17421)
 - **Vídeo**, sobre ese paper, del canal de YouTube [1littlecoder](https://www.youtube.com/@1littlecoder): [GPT-4 Vision PROMPT ENGINEERING Tutorial](https://www.youtube.com/watch?v=dtKBAST12O0)

## Reentrenamiento de LLMs (fine-tuning)

Tutoriales:

 - [Fine-Tuning LLaMA 2: A Step-by-Step Guide to Customizing the Large Language Model](https://www.datacamp.com/tutorial/fine-tuning-llama-2): <q>Learn how to fine-tune Llama-2 on Colab using new techniques to overcome memory and computing limitations to make open-source large language models more accessible.</q>. <q>Updated Oct 2023  · 12 min read</q>. Author: Abid Ali Awan. Usa un conjunto de datos ya existente (<q>We will load the “guanaco-llama2-1k” dataset from the Hugging Face hub. The dataset contains 1000 samples and has been processed to match the Llama 2 prompt format, and is a subset of the excellent timdettmers/openassistant-guanaco dataset.</q>), pero muestra el formato que cada prompt debe tener, por lo que es posible usar esto con uno personal. La sección 9 de ese tutorial es la del entrenamiento propiamente dicho (Todo su texto: <q>9. Model fine-tuning</q>: <q>Supervised fine-tuning (SFT) is a key step in reinforcement learning from human feedback (RLHF). The TRL library from HuggingFace provides an easy-to-use API to create SFT models and train them on your dataset with just a few lines of code. It comes with tools to train language models using reinforcement learning, starting with supervised fine-tuning, then reward modeling, and finally, proximal policy optimization (PPO).</q>. <q>We will provide SFT Trainer the model, dataset, Lora configuration, tokenizer, and training parameters.</q>. <q>We will use .train() to fine-tune the Llama 2 model on a new dataset. It took one and a half hours for the model to complete 1 epoch.</q>. <q>After training the model, we will save the model adopter and tokenizers. You can also upload the model to Hugging Face using a similar API.</q>). **Para probarlo en 'Colab' es necesario pagar por una GPU con más memoria** (a día 2024-05-11, la versión gratuita ofrece una GPU T4 con 15.0 GB de RAM, además de 12.7 GB de RAM de sistema, que para esto da igual (es decir, esos 12.7 GB son más que suficientes, ya que para esto sólo usa 2.7 GB)). Todo el código seguido:
<pre>
%%capture
%pip install accelerate peft bitsandbytes transformers trl

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

# Model from Hugging Face hub
base_model = "NousResearch/Llama-2-7b-chat-hf"

# New instruction dataset
guanaco_dataset = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model
new_model = "llama-2-7b-chat-guanaco"

dataset = load_dataset(guanaco_dataset, split="train")

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

# Train model
trainer.train()

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

from tensorboard import notebook
log_dir = "results/runs"
notebook.start("--logdir {} --port 4000".format(log_dir))

logging.set_verbosity(logging.CRITICAL)

prompt = "Who is Leonardo Da Vinci?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"&lt;s&gt;[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

prompt = "What is Datacamp Career track?"
result = pipe(f"&lt;s&gt;[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
</pre>
 - Vídeotutoriales de 'YouTube' que usan [unsloth](https://github.com/unslothai/unsloth):
   - [Fine-tuning a Phi-3 LeetCode Expert? - Dataset Generation, Unsloth ++](https://www.youtube.com/watch?v=DeuyD-ZA-58). Canal: [All About AI](https://www.youtube.com/@AllAboutAI)
   - Del canal: [Nodematic Tutorials](https://www.youtube.com/@nodematic):
     - [Fine-Tuning Gemma (Easiest Method with Unsloth & Colab)](https://www.youtube.com/watch?v=pWZfufhF45o).
     - [Llama 3 Fine Tuning for Dummies (with 16k, 32k,... Context)](https://www.youtube.com/watch?v=3eq84KrdTWY)
