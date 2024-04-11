import argparse
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.models.opt.modeling_opt import *


def generate(task_info, device, model, tokenizer):
    contexts = task_info["prompt_seqs"]
    inputs = tokenizer(contexts, return_tensors="pt").to(device)
    print(f"start_ids: length ({inputs.input_ids.shape[0]}) ids: {inputs.input_ids}")
    input_length = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs, do_sample=True, top_p=task_info['top_p'],
        temperature=1.0, top_k=task_info['top_k'],
        max_new_tokens=task_info["output_len"],
        num_beams=task_info['beam_width'],
        num_return_sequences = task_info['beam_width'],
        return_dict_in_generate=True,
        output_scores=True,  # return logit score
        output_hidden_states=False,  # return embeddings
    )
    print(f"[INFO] raw output: {outputs.keys()} {len(outputs)}, {outputs[0].shape},  ({outputs[1][0].shape},{outputs[1][1].shape}) {len(outputs[2])}")
    for i in range(task_info['beam_width']):
        token = outputs.sequences[i, input_length:]  # exclude context input from the output
        print(f"[INFO] raw token: {token}")
        output = tokenizer.decode(token)
        print(f"[INFO][Context]: {contexts}\n[Output-{i+1}]\n{output}\n")


def test_model(args):
    print(f"<test_model> initialization start")
    device = torch.device(args.get('device', 'cpu'))
    tokenizer = AutoTokenizer.from_pretrained(args['hf_model_name'])
    model = AutoModelForCausalLM.from_pretrained(args['hf_model_name'])
    model = model.to(device)
    torch.manual_seed(0)
    task_info = {
        "seed": 0,
        "prompt_seqs": args.get('prompt_seq', 'what is Apple?'),
        "output_len": args.get('output_len', 20),
        "beam_width": args.get('beam_width', 1),
        "top_k": args.get('top_k', 50),
        "top_p": args.get('top_p', 1.0),
        "beam_search_diversity_rate": 0,
        "len_penalty": 0,
        "repetition_penalty": 1.0,
        "stop": args.get("stop", []),
    }
    print(f"<test_model> initialization done")
    generate(task_info, device, model, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_model_name', type=str, default='facebook/opt-350m',
                        help='hugging face model name (used to load model).')
    parser.add_argument('--prompt_seq', type=str, default='What is Apple?',
                        help='-')
    parser.add_argument('--output_len', type=int, default=20,
                        help='-')
    parser.add_argument('--top_k', type=int, default=50,
                        help='-')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='-')
    parser.add_argument('--beam_width', type=int, default=1,
                        help='-')
    
    args = parser.parse_args()
    test_model(args={
        "hf_model_name": args.hf_model_name,
        "prompt_seq": args.prompt_seq,
        "output_len": args.output_len,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "beam_width": args.beam_width,
        "device": "cpu",
        "dtype": torch.float32,
    })