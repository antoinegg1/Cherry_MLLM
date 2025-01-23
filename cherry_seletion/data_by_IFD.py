import torch
import json
import numpy as np
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import LlavaNextForConditionalGeneration,LlavaNextProcessor
'''
python /mnt/file2/changye/Cherry_MLLM/cherry_seletion/data_by_IFD.py \
    --pt_data_path /mnt/file2/changye/Cherry_MLLM/Cheery_cherry.pt \
    --data_path data/alpaca_data.json \
    --model_name_or_path /mnt/file2/changye/model/llava \
    --save_path /mnt/file2/changye/dataset/AA_preference_Cherry_cherry_sample \
    --max_length 4096 \
    --sample_rate 0.1 \
    --prompt alpaca
'''

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
    parser.add_argument("--pt_data_path", type=str, default='')
    parser.add_argument("--data_path", type=str, default='')
    parser.add_argument("--save_path", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str, default='')
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--sample_rate", type=float, default=0.1)
    parser.add_argument("--sample_number", type=int, default=0)
    parser.add_argument("--prompt", type=str, default='alpaca', help='wiz, alpaca')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    print(args)

    processor=LlavaNextProcessor.from_pretrained(args.model_name_or_path)

    pt_data = torch.load(args.pt_data_path)
    combine_pt_data=[]
    for i in range(len(pt_data)):
        combine_pt_data.extend(pt_data[i])
    data = load_dataset('PKU-Alignment/align-anything',name='text-image-to-text',cache_dir="/mnt/file2/changye/dataset/AA_preference")['train']

    mean_rate_list = []
    mean_list_1 = []
    mean_list_2 = []
    for i in tqdm(range(len(combine_pt_data))):

        pt_data_i = combine_pt_data[i]
        loss_1_list = pt_data_i['token_loss'][1]
        loss_2_list = pt_data_i['token_loss'][2]

        data_i = data[i]
        instruct_i = data_i['question']
        p_response_i = data_i['p_response']
        output_i = data_i[f'response_{p_response_i}']

        direct_answer_text = '### Response:' + output_i
        if args.prompt == 'wiz':
            whole_text = instruct_i+'\n\n### Response:'+output_i
        elif args.prompt == 'alpaca':
            input_i =''
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

        # Tokenize the input text
        instruct_i_input_ids = processor(instruct_i, return_tensors="pt", truncation=True, max_length=args.max_length)['input_ids'].to('cpu')
        # breakpoint()
        instruct_i_len = instruct_i_input_ids.shape[1] 

        def get_loss_part_text(processor, text, target_span, max_length, loss_list_):

            input_ids = processor(text, return_tensors="pt", truncation=True, max_length=max_length)['input_ids'].to('cpu')
            start_index = text.rfind(target_span)
            text_temp = text[:start_index]
            token_id_temp = processor(text_temp)['input_ids']
            start_token = len(token_id_temp) 
            end_token_real = input_ids.shape[1]

            loss_list = loss_list_[start_token-1:end_token_real-1] 

            return end_token_real - start_token , input_ids[0][start_token:end_token_real], np.array(loss_list)
        
        if args.max_length-instruct_i_len > 0:

            len_1, token_ids_1, loss_list_1 = get_loss_part_text(processor, direct_answer_text, output_i, args.max_length-instruct_i_len+4, loss_1_list)
            len_2, token_ids_2, loss_list_2 = get_loss_part_text(processor, whole_text, output_i, args.max_length, loss_2_list)

            if len_1 <= 0 or len_2 <= 0:
                continue

            if instruct_i_len + len_1 > args.max_length:
                continue

            mean_1 = loss_list_1.mean()
            mean_2 = loss_list_2.mean()
            mean_rate = mean_2/mean_1
            # if mean_rate > 1: 
            #     continue

            mean_rate_list.append((mean_rate,i))
            mean_list_1.append((mean_1,i))
            mean_list_2.append((mean_2,i))

        else:
            continue

    print('Do Rate')
    mean_rate_list = sorted(mean_rate_list)
    if args.sample_number == 0:
        args.sample_number = int(len(mean_rate_list)*args.sample_rate)
    # 将数据按照sample-rate 切分全部保存
    sample_bound=[10,9,8,7,6,5,4,3,2,1]
    segment_size = len(mean_rate_list) // 10
    # breakpoint()
    for i in sample_bound:
        start_index = segment_size * (i - 1)  # 计算当前段的起始索引
        if i == 10:
            end_index = len(mean_rate_list)  # 最后一段包含剩余的所有数据
        else:
            end_index = segment_size * i  # 当前段的结束索引

        # 选择当前段的数据
        mean_rate_list_id = list(range(start_index, end_index))
        mean_rate_list_id_sample = [mean_rate_list[id][1] for id in mean_rate_list_id]

        new_data =data.select(mean_rate_list_id_sample)
        print('New data len \n',len(new_data))
        new_data.save_to_disk(args.save_path+'_'+str(i)+'_'+str(i//args.sample_number))
    
    
    
    # mean_rate_list_id = [i for i in range(len(mean_rate_list))][-args.sample_number:]
    # mean_rate_list_id_sample = [mean_rate_list[id][1] for id in mean_rate_list_id]


    # new_data =data.select(mean_rate_list_id_sample)
    # print('New data len \n',len(new_data))
    # new_data.save_to_disk(args.save_path)


if __name__ == '__main__':
    main()