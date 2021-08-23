#!/usr/bin/env python3

from configs import *


get_ipython().run_cell_magic('time', '', 'from pathlib import Path\n\nfrom tokenizers import ByteLevelBPETokenizer\n\npaths = [str(x) for x in Path(".").glob("**/*.txt")]\n\n# Initialize a tokenizer\ntokenizer = ByteLevelBPETokenizer()\n\n# Customize training\ntokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[\n    "<s>",\n    "<pad>",\n    "</s>",\n    "<unk>",\n    "<mask>",\n])')



get_ipython().system(f'mkdir {TOKENIZER_DIR}')
tokenizer.save_model(TOKENIZER_DIR)


