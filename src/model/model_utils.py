

model_dirs = {
    'mistral0.1': 'mistralai/Mistral-7B-Instruct-v0.1',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
    'bloomz': 'bigscience/bloomz-7b1',
    'llama3.1': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'qwen': 'Qwen/Qwen1.5-7B-Chat',
    'gemma2': 'google/gemma-2-9b-it',
    'gptj': 'EleutherAI/gpt-j-6b'
}


def load_model(args, model_name=None, peft_path=None):
    model_name = args.model if model_name is None else model_name
    
    if model_name == 'mistral0.1' :
        from model.mistral import MistralWrapper
        return MistralWrapper(args, model_dirs['mistral0.1'], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    if model_name == 'mistral' :
        from model.mistral import MistralWrapper
        return MistralWrapper(args, model_dirs['mistral'], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    elif model_name == 'bloomz' :
        from model.bloomz import BloomzWrapper
        return BloomzWrapper(args, model_dirs['bloomz'], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    elif model_name == 'llama3.1' :
        from model.llama3 import Llama3Wrapper
        return Llama3Wrapper(args, model_dirs['llama3.1'], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    elif model_name == 'qwen' :
        from model.qwen import QwenWrapper
        return QwenWrapper(args, model_dirs['qwen'], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    elif model_name == 'gemma2' :
        from model.gemma import GemmaWrapper
        return GemmaWrapper(args, model_dirs['gemma2'], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
    elif model_name == 'gptj' :
        from model.gptj import GPTJWrapper
        return GPTJWrapper(args, model_dirs['gptj'], memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb, lora_adapter_path=peft_path)
           

def sanitize_output(args, gen_responses, model_name, do_strip=True):

    for i, gen_response in enumerate(gen_responses):
        if model_name == 'mistral':
            gen_response = gen_response.replace('▁',' ').replace('<SYS>','').replace('<0x0A><0x0A>',' ').replace('<0x0A>','').replace('</s>', '')
        elif model_name == 'bloomz':
            gen_response = gen_response.split('</s>')[0].replace('Ġ',' ').replace('ĊĊ','')
        elif model_name =='llama3.1':
            import pdb;pdb.set_trace()
            gen_response = gen_response.split('<|eot_id|>')[0].replace('Ġ',' ').replace('âĢĻ',"'").replace('ĊĊ','').replace('Ċ',' ')
        elif model_name == 'qwen':
            gen_response = gen_response.split('</s>')[0].replace('Ġ',' ').replace('ĊĊ','')
        elif model_name == 'gemma2':
            gen_response = gen_response.replace('▁',' ').replace('<SYS>','').replace('<0x0A><0x0A>',' ').replace('<0x0A>','').replace('</s>', '')
        elif model_name == 'gptj':
            gen_response = gen_response.split('</s>')[0].replace('Ġ',' ').replace('ĊĊ','')
        
        gen_responses[i] = gen_response.strip() if do_strip else gen_response
    return gen_responses
