import pandas as pd
import random


def load_data(args, tokenizer, model_name, split=None):
    if args.data == 'dro_data':
        from data.dro_data import load_data as load_drodata
        return load_drodata(args, tokenizer, split='train')
    elif args.data == 'advbench':
        from data.advbench import load_data as load_advbench
        return load_advbench(args, tokenizer, model_name, split='train')

def format_input(args, query, response, tokenizer, model_name, dialog=False):
    if model_name == 'bloomz' :
        return f'<s>\n\nUSER: {query} \nASSISTANT: {response}</s>' if dialog else f'<s>\n\nUSER: {query} \nASSISTANT: '
    elif model_name == 'gptj':
        return f"<s>[INST] {query} [/INST] {response}</s>" if dialog else f"<s>[INST] {query} [/INST]"
    inp = [{'role': 'user', 'content': query},{'role': 'assistant', 'content': response}] if dialog else [{'role': 'user', 'content': query}]
    return tokenizer.apply_chat_template(inp, tokenize=False, add_generation_prompt=True)
    

    
def batchify(data_list, batch_size):
    batches = []
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        batch = pd.DataFrame(batch).to_dict('list')
        batches.append(batch)
    return batches

