from datasets import load_dataset, Dataset
import random
import numpy as np
import pandas as pd
from copy import deepcopy

from data.data_utils import *

def load_data(args, tokenizer, model_name, split):
    
    # NOTE: advbench is technically a csv dataset, but I use a hfhub mirror since this makes dataset management easier
    dataset = pd.DataFrame(load_dataset('walledai/AdvBench', split="train", cache_dir=args.data_dir))
    
    dataset['input'] = [format_input(args, query, resp[0], tokenizer, model_name, dialog=False) for query, resp in zip(dataset['prompt'], dataset['target'])]
    dataset['dialog'] = [format_input(args, query, resp[0], tokenizer, model_name, dialog=True) for query, resp in zip(dataset['prompt'], dataset['target'])]
    
    dataset = dataset.to_dict('records')
    dataset = batchify(dataset, args.batch_size)
    return dataset