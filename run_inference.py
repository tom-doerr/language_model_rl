#!/usr/bin/env python3

from configs import *
from transformers import pipeline


if LM_TO_USE == 'roberta': 
    fill_mask = pipeline(
        "fill-mask",
        model=TOKENIZER_DIR,
        tokenizer=TOKENIZER_DIR
    )

    # episode_str = 'obs: <mask>'
    episode_str = 'obs: -0.0 -0.0 -0.0 0.0  reward: 1.0  action: <mask>'
    
    for i in range(1):
        completion = fill_mask(episode_str)
        episode_str = completion[0]['sequence'] + '<mask>'

    print("episode_str:", episode_str)
    # completion = fill_mask("obs: <mask>")
    # print("completion:", completion)

elif LM_TO_USE == 'gptneo':
    text_generation = pipeline(
        "text-generation",
        model=TOKENIZER_DIR,
        tokenizer=TOKENIZER_DIR
    )

    completion = text_generation('obs:')
    print("completion:", completion)


# fill_mask("Jen la komenco de bela <mask>.")

