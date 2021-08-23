#!/usr/bin/env python3

from configs import *
from transformers import RobertaConfig, GPTNeoConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM, GPTNeoForCausalLM


if LM_TO_USE == 'roberta': 
    config = RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_DIR, max_len=512)
    model = RobertaForMaskedLM(config=config)
elif LM_TO_USE == 'gptneo':
    config = GPTNeoConfig(
    )
    tokenizer = RobertaTokenizerFast.from_pretrained(TOKENIZER_DIR, max_len=512)
    model = GPTNeoForCausalLM(config=config)





model.num_parameters()

# get_ipython().run_cell_magic('time', '', 'from transformers import LineByLineTextDataset\n\ndataset = LineByLineTextDataset(\n    tokenizer=tokenizer,\n    file_path="./data/rollouts.txt",\n    block_size=128,\n)')

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=DATA_FILE,
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=TOKENIZER_DIR,
    overwrite_output_dir=True,
    num_train_epochs=1,
    # per_gpu_train_batch_size=64,
    per_gpu_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

get_ipython().run_cell_magic('time', '', 'trainer.train()')
trainer.save_model(TOKENIZER_DIR)
