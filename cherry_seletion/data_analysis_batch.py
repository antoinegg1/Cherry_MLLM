import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from datasets import load_from_disk, load_dataset
import torch.nn as nn
import time
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction='none')
to_pil=transforms.ToPILImage()
transform = transforms.ToTensor()
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
'''
python /mnt/file2/changye/Cherry_MLLM/cherry_seletion/data_analysis_batch.py \
    --data_path /mnt/file2/changye/dataset/Align-Anything_preference/text-image-to-text/train \
    --save_path /mnt/file2/changye/dataset/Align-Anything_preference_pre.pt \
    --model_name_or_path /mnt/file2/changye/model/llava \
    --max_length 4096 \
    --prompt alpaca \
    --mod pre \
    --batch_size 8
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
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--prompt", type=str, default='wiz', help='wiz, alpaca')
    parser.add_argument("--mod", type=str, default='pre', help='pre, cherry')
    args = parser.parse_args()
    return args

class CustomDataset(Dataset):
    def __init__(self, data, processor, max_length, prompt_type, mod):
        self.data = data
        self.processor = processor
        self.max_length = max_length
        self.prompt_type = prompt_type
        self.mod = mod

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_i = self.data[idx]
        instruct_i = data_i['question'].replace('<image>',' ')
        instruct_i = f"<image> {instruct_i}"
        image_i = data_i['image']
        image_i=transform(image_i.resize((336,336)).convert('RGB'))
        p_response_i = data_i['p_response']
        output_i = data_i[f'response_{p_response_i}']

        direct_answer_text = '### Response:' + output_i
        if self.prompt_type == 'wiz':
            whole_text = instruct_i + '\n\n### Response:' + output_i
            input_i = data_i['input'] if 'input' in data_i else ''
            if input_i != '':
                whole_text = instruct_i + '\nInput:' + input_i + '\n\n### Response:' + output_i

        elif self.prompt_type == 'alpaca':
            input_i = data_i['input'] if 'input' in data_i else ''
            if input_i == '':
                temp_dict = {'instruction': instruct_i}
                promt_to_use = PROMPT_DICT["prompt_no_input"].format_map(temp_dict)
                whole_text = promt_to_use + output_i
                instruct_i = promt_to_use
            else:
                temp_dict = {'instruction': instruct_i, 'input': input_i}
                promt_to_use = PROMPT_DICT["prompt_input"].format_map(temp_dict)
                whole_text = promt_to_use + output_i
                instruct_i = promt_to_use

        return instruct_i, image_i, direct_answer_text, whole_text, output_i

def get_perplexity_and_embedding_whole_text_batch(processor, model, texts, images, max_length):
    images = [to_pil(image) for image in images]
    texts=[text for text in texts]
    # breakpoint()
    inputs = processor(text=texts, images=images, return_tensors="pt", truncation=True, max_length=max_length, padding=True,do_rescale=False).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pixel_values=inputs['pixel_values'],
            image_sizes=inputs['image_sizes'],
            labels=inputs['input_ids'].clone().contiguous()
        )
    # breakpoint()
    loss = outputs.loss
    perplexity=[]
    for i in range(len(loss)):
        perplexity.append(torch.exp(loss[i]).to('cpu'))
    hidden_states = outputs.logits
    embeddings = hidden_states[:,-1,:]
    sentence_embeddings = embeddings.mean(dim=1)

    return perplexity, sentence_embeddings.to('cpu')

def get_perplexity_and_embedding_part_text_batch(processor, model, texts, target_spans, max_length, images=None):
    if images is None:
        texts=[text for text in texts]
        inputs = processor(text=texts, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
    else:
        images=[to_pil(image) for image in images]
        texts=[text for text in texts]
        inputs = processor(images, texts, return_tensors="pt", truncation=True, max_length=max_length, padding=True,do_rescale=False).to(device)

    input_ids = inputs['input_ids']
    start_tokens = [text.rfind(target_span) for text, target_span in zip(texts, target_spans)]
    start_tokens = [len(processor.tokenizer.encode(text[:start_idx])) for text, start_idx in zip(texts, start_tokens)]
    end_tokens = input_ids.shape[1]

    labels = input_ids.clone()
    for i, start_token in enumerate(start_tokens):
        labels[i, :start_token] = -100

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=inputs['attention_mask'],
            pixel_values=inputs['pixel_values'],
            image_sizes=inputs['image_sizes'],
            labels=labels
        )

    loss = outputs.loss
    perplexity=[]
    for i in range(len(loss)):
        perplexity.append(torch.exp(loss[i]).to('cpu'))

    losses = []
    logits = outputs.logits
    for i in range(1, end_tokens):
        breakpoint() # wait for check
        log_prob_dist = log_softmax(logits[:, i-1])
        true_tokens = input_ids[:, i]
        token_loss = nll_loss(log_prob_dist, true_tokens)
        losses.append(token_loss.mean().item())

    return perplexity, losses


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

    data = load_dataset('PKU-Alignment/align-anything', name='text-image-to-text', cache_dir="/mnt/file2/changye/dataset/AA_preference")['train']
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    sampled_data = data.select(list(range(start_idx, end_idx)))

    dataset = CustomDataset(sampled_data, processor, args.max_length, args.prompt, args.mod)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    strat_time = time.time()
    new_data = []

    for batch in tqdm(dataloader):
        instruct_batch, image_batch, direct_answer_batch, whole_text_batch, output_batch = batch

        if args.mod == 'pre':
            ppl_batch, emb_batch = get_perplexity_and_embedding_whole_text_batch(processor, model, instruct_batch, image_batch, args.max_length)
            breakpoint()
            for i in range(len(ppl_batch)):
                temp_data_i = {'ppl': [ppl_batch[i], 0, 0], 'sent_emb': [emb_batch[i], 0, 0]}
                new_data.append(temp_data_i)

        elif args.mod == 'cherry':
            ppl_out_alone_batch, loss_list_alone_batch = get_perplexity_and_embedding_part_text_batch(processor, model, direct_answer_batch, output_batch, args.max_length // 2)
            ppl_out_condition_batch, loss_list_condition_batch = get_perplexity_and_embedding_part_text_batch(processor, model, whole_text_batch, output_batch, args.max_length, images=image_batch)

            for i in range(len(ppl_out_alone_batch)):
                temp_data_i = {
                    'ppl': [0, ppl_out_alone_batch[i], ppl_out_condition_batch[i]],
                    'token_loss': [[], loss_list_alone_batch[i], loss_list_condition_batch[i]]
                }
                new_data.append(temp_data_i)

    print('New data len:', len(new_data))
    torch.save(new_data, args.save_path)
    print('Time Used:', (time.time() - strat_time) / 60, '(min)')

if __name__ == "__main__":
    main()
