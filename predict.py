"""Load base + LoRA and reuse training prompts; shared by standalone evaluation."""
import argparse
import json
from data_utils import SYSTEM_PROMPT, project_path, user_prompt


def load_adapter(model_path, adapter_path, device='auto', load_in_4bit=False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    base, adapter = project_path(model_path), project_path(adapter_path)
    run = json.loads((adapter.parent / 'run_config.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    cpu = device == 'cpu'
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError('没有 CUDA；使用 --device cpu 或安装 CUDA 版 torch')
    dtype = torch.float32 if cpu else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    kwargs = {'torch_dtype': dtype, 'local_files_only': True, 'attn_implementation': 'sdpa'}
    if load_in_4bit:
        if cpu:
            raise ValueError('4-bit 推理需要 CUDA')
        kwargs.update(device_map={'': 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype))
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    model = PeftModel.from_pretrained(model, adapter)
    if not load_in_4bit:
        model.to('cpu' if cpu else 'cuda')
    model.eval()
    model.config.use_cache = True
    return model, tokenizer, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--adapter-path', required=True)
    parser.add_argument('--code-file', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--device', choices=['auto', 'cpu'], default='auto')
    args = parser.parse_args()
    import torch
    model, tokenizer, run = load_adapter(args.model_path, args.adapter_path, args.device, args.load_in_4bit)
    code = project_path(args.code_file).read_text(encoding='utf-8')
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_prompt(code, run['line_format'])}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                           return_dict=True, return_tensors='pt').to(model.device)
    if inputs['input_ids'].shape[1] + args.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError('输入 + 输出预算超过模型上下文上限')
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    print(tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True))


if __name__ == '__main__':
    main()
