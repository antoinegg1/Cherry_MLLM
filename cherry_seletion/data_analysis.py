import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration,LlavaNextProcessor
from datasets import load_from_disk,load_dataset
import torch.nn as nn
import time
log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction='none')
# CUDA_VISIBLE_DEVICES = 1,2,3,4,5,6,7
'''
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python /mnt/file2/changye/Cherry_MLLM/cherry_seletion/data_analysis.py \
    --data_path /mnt/file2/changye/dataset/Align-Anything_preference/text-image-to-text/train \
    --save_path /mnt/file2/changye/dataset/Align-Anything_preference_pre \
    --model_name_or_path /mnt/file2/changye/model/llava \
    --max_length 4096 \
    --prompt alpaca \
    --mod pre
'''


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default='wiz', help='wiz, alpaca')
    parser.add_argument("--mod", type=str, default='pre', help='pre, cherry')
    args = parser.parse_args()
    return args

# Used to get the ppl and emb for the whole input
def get_perplexity_and_embedding_whole_text(processor, model, text,image_i, max_length):
    inputs=processor( image_i,text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    with torch.no_grad(): 
        outputs = model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        pixel_values=inputs['pixel_values'],
                        image_sizes=inputs['image_sizes'],
                        labels=inputs['input_ids'].clone().contiguous()
                        )
    loss = outputs.loss
    perplexity = torch.exp(loss)
    # breakpoint()
    hidden_states = outputs.logits
    embeddings = hidden_states[-1]
    sentence_embedding = embeddings.mean(dim=1)

    return perplexity.to('cpu'), sentence_embedding.to('cpu')

# Used to get the ppl and emb for part of input, used in conditional version, and token-wise loss
def get_perplexity_and_embedding_part_text(processor, model, text, target_span, max_length,image=None):
    if image is None:
        inputs=processor(text=text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    else:
        inputs=processor(image,text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    input_ids=inputs['input_ids']
    start_index = text.rfind(target_span)
    start_token = len(processor.tokenizer.encode(text[:start_index]))
    end_token = input_ids.shape[1]

    labels = input_ids.clone()
    labels[0, :start_token] = -100

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=inputs['attention_mask'],
            pixel_values=inputs['pixel_values'],
            image_sizes=inputs['image_sizes'],
            labels=labels
            )

    loss = outputs.loss
    perplexity = torch.exp(loss)

    losses = []
    logits = outputs.logits
    for i in range(1, end_token):
        log_prob_dist = log_softmax(logits[0, i-1])
        true_token = input_ids[0, i]
        token_loss = nll_loss(log_prob_dist.unsqueeze(0), true_token.unsqueeze(0))
        losses.append(token_loss.item())

    return perplexity.to('cpu'), 0, losses


def main():

    args = parse_args()
    print(args)

    model = LlavaNextForConditionalGeneration.from_pretrained(args.model_name_or_path, device_map="auto", cache_dir='../cache')
    processor = LlavaNextProcessor.from_pretrained(args.model_name_or_path, cache_dir='../cache')

    model.eval()

    if args.save_path[-3:] != '.pt':
        args.save_path += '.pt'
    if os.path.exists(args.save_path):
        print('save_path exists!')
        raise Exception

    data = load_dataset('PKU-Alignment/align-anything',name='text-image-to-text',cache_dir="/mnt/file2/changye/dataset/AA_preference")['train']
    # breakpoint()
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    sampled_data = data.select(list(range(start_idx,end_idx)))


    strat_time = time.time()
    new_data = []
    for i in tqdm(range(len(sampled_data))):

        data_i = sampled_data[i]
        instruct_i = data_i['question'].replace('<image>',' ')
        instruct_i=f"<image> {instruct_i}"
        image_i= data_i['image']
        p_response_i = data_i['p_response']
        output_i = data_i[f'response_{p_response_i}']

        direct_answer_text = '### Response:' + output_i
        if args.prompt == 'wiz':
            whole_text = instruct_i+'\n\n### Response:'+output_i
            input_i = data_i['input'] if 'input' in data_i.keys() else ''
            if input_i != '':
                whole_text = instruct_i+'\nInput:'+input_i+'\n\n### Response:'+output_i

        elif args.prompt == 'alpaca':
            input_i = data_i['input'] if 'input' in data_i.keys() else ''
            if input_i == '':
                temp_dict = {'instruction':instruct_i}
                promt_to_use = PROMPT_DICT["prompt_no_input"].format_map(temp_dict)
                whole_text = promt_to_use + output_i
                instruct_i = promt_to_use
            else:
                temp_dict = {'instruction':instruct_i,'input':input_i}
                promt_to_use = PROMPT_DICT["prompt_input"].format_map(temp_dict)
                whole_text = promt_to_use + output_i
                instruct_i = promt_to_use

        temp_data_i = {}
        if args.mod == 'pre':
            ppl_ins_alone, emb_ins_alone = get_perplexity_and_embedding_whole_text(processor, model, instruct_i,image_i, args.max_length)
            temp_data_i['ppl'] = [ppl_ins_alone,0,0]
            temp_data_i['sent_emb'] = [emb_ins_alone,0,0]

        elif args.mod == 'cherry':
            # inputs=processor(instruct_i,image_i,return_tensors="pt",truncation=True,max_length=args.max_length).to(device)
            # instruct_i_input_ids =inputs['input_ids']
            # instruct_i_len = instruct_i_input_ids.shape[1]
        
            ppl_out_alone, _, loss_list_alone = get_perplexity_and_embedding_part_text(processor, model, direct_answer_text, output_i, args.max_length//2)
            ppl_out_condition, _, loss_list_condition = get_perplexity_and_embedding_part_text(processor, model, whole_text, output_i, args.max_length,image=image_i)

            temp_data_i['ppl'] = [0,ppl_out_alone,ppl_out_condition]
            temp_data_i['token_loss'] = [[],loss_list_alone,loss_list_condition]

        new_data.append(temp_data_i)
        pass

    print('New data len:', len(new_data))
    torch.save(new_data,args.save_path)

    print('Time Used:',(time.time()-strat_time)/60,'(min)')

if __name__ == "__main__":
    main()