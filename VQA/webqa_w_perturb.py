"""
For a given category, generate answers on original training image + perturbed image and store them -
{
    <guid>: {"A": <ans>, "llava_A": <ans>, "llava_A_perturbed": [<ans1> , <ans2> ...]}
}
"""
import argparse
import os
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from webqa_test import get_multihop_image_prompt, get_multihop_text_prompt, get_si_prompt, get_single_text_prompt, get_ques_prompt, read_image, get_remaining_ids
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
import json
from io import BytesIO
import base64
import os.path as osp
warnings.filterwarnings("ignore")


dataset_dir = '/mnt/disks/data/webqa/WebQA_train_val.json'
lineidx_dir = '/mnt/disks/data/webqa/imgs.lineidx'
images_dir = '/mnt/disks/data/webqa/imgs.tsv'
disk_root = "/mnt/disks/data/webqa"
train_ptb_data_dir = osp.join(disk_root, "perturbed", "train")
val_ptb_data_dir = osp.join(disk_root, "perturbed", "val")
lineidx_dir = '/mnt/disks/data/webqa/imgs.lineidx'
images_dir = '/mnt/disks/data/webqa/imgs.tsv'

pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
        "multimodal": True,
    }
overwrite_config = {}
overwrite_config["image_aspect_ratio"] = "pad"
llava_model_args["overwrite_config"] = overwrite_config

def read_perturbed_img(image_id, qid, sample_id, ptb_data_dir):
    img_name = f"{image_id}_{qid}_{sample_id}.jpeg"
    img_path = osp.join(ptb_data_dir, img_name)
    im = Image.open(img_path)
    if im.mode != 'RGB':
        im = im.convert('RGB')  # Convert to RGB if not already
    return im

def read_data(split, qcate):
    with open(dataset_dir, 'r') as f:
        data = json.load(f)
    val_data = {}
    train_data = {}
    for id,dict in data.items():
        if dict['Qcate'] == 'text':
            continue
        elif not len(dict['img_posFacts']) > 0:
            continue
        elif dict['split'] == 'val':
            val_data[id] = dict
        elif dict['split'] == 'train':
            train_data[id] = dict
    if qcate == 'all':
        return train_data if split == 'train' else val_data

    cat_val_data = {}
    for id,d in val_data.items():
        if d['Qcate'] == qcate: 
            cat_val_data[id] = d
            
    cat_train_data = {}
    for id,d in train_data.items():
        if d['Qcate'] == qcate:
            cat_train_data[id] = d
    return train_data if split == 'train' else val_data

def get_ans(qid, data, model, image_processor, tokenizer, device, sample_idx, use_ptb_img = False):
    ques = data['Q']
    images = []
    titles = []
    texts = []
    for img in data['img_posFacts']:
        image_id = img['image_id']
        # We only have the first perturbed image which should be enough to change the answer
        if use_ptb_img and len(images) == 0:
            images.append(read_perturbed_img(str(image_id), qid, str(sample_idx)))
        else:
            images.append(read_image(image_id))
        titles.append(img['caption'])

    for txt in data['txt_posFacts']:
        texts.append(txt['fact'])
        
    # skip if no images
    if len(images) == 2:
        promptImg = get_multihop_image_prompt(titles)
    elif len(images) == 1:
        promptImg = get_si_prompt(titles)
    else:
        return ""

    promptTxt = ""
    if len(texts) == 2:
        promptTxt = get_multihop_text_prompt(texts)
    elif len(texts) == 1:
        promptTxt = get_single_text_prompt(texts)
    
    prompt = promptImg + promptTxt + get_ques_prompt(ques)
    image_tensors = process_images(images, image_processor, model.config)
    image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]
    
    conv_template = "qwen_1_5"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size for image in images]
    # Generate response
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    return text_outputs[0]

def get_num_ptbs(id, d, split):
    ptb_files = os.listdir(train_ptb_data_dir)
    if split == 'val':
        ptb_files = os.listdir(val_ptb_data_dir)
    elif split == 'full':
        ptb_files += os.listdir(val_ptb_data_dir)

    return len([i for i in ptb_files if i.startswith(f"{id}_{d}")])
        
def main():
    args = argparse.ArgumentParser()
    args.add_argument("split", type=str, choices=["train", "val", "full"], default="val")
    args.add_argument("qcate", type=str, choices=["color", "shape", "yesno", "number", "all"], default="color")
    args.parse_args()
    out_file_name = f"llavanext_webqa_ptb_{args.split}_{args.qcate}.json"
    out_file = osp.join(disk_root, out_file_name)
    data = read_data(args.split, args.qcate)
    remaining_ids = get_remaining_ids(out_file)
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, load_8bit=True, **llava_model_args)
    model.eval()
    f = open(out_file, 'a')
    results = {} # We accumulate results in memory to keep things simple
    count = 0
    for id,d in data.items():
        if len(remaining_ids) > 0 and id not in remaining_ids:
            continue
        try:
            count += 1
            if count == 5:
                break
            ans_orig = get_ans(id, d, model, image_processor, tokenizer, max_length, device, use_ptb_img=False)
            # List of answers for each ptb
            num_ptbs = get_num_ptbs(id, d, args.split)
            ans_ptb = []
            for i in range(num_ptbs):
                ans_ptb.append(get_ans(id, d, model, image_processor, tokenizer, max_length, device, sample_idx=i, use_ptb_img=True))
            results[id] = {}
            results[id]['A'] = d['A']
            results[id]['llava_A'] = ans_orig
            results[id]['llava_A_perturbed'] = ans_ptb
            json.dump(results, f)
        except Exception as e:
            print(e)
            continue
    f.close()

if __name__ == '__main__':
    main()