from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
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
warnings.filterwarnings("ignore")

dataset_dir = '/mnt/disks/data/webqa/WebQA_train_val.json'
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

with open(dataset_dir, 'r') as f:
    data = json.load(f)

val_data = {}
for id,dict in data.items():
    if dict['Qcate'] == 'text':
        continue
    elif not dict['split'] == 'val':
        continue
    elif not len(dict['img_posFacts']) > 0:
        continue
    val_data[id] = dict


with open(lineidx_dir, "r") as fp_lineidx:
    lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]

def get_multihop_prompt(ques, titles, captions):
    question = f"{DEFAULT_IMAGE_TOKEN} This is the first image.\nTitle: {titles[0]}\n\nNow, let's look at another image: {DEFAULT_IMAGE_TOKEN} Title: {titles[1]}\nRespond with the correct answer to the following question in a complete sentence. Question: {ques}"
    return question

def get_si_prompt(ques, titles, captions):
    question = f"{DEFAULT_IMAGE_TOKEN} You have to answer a question about this image\nTitle: {titles[0]}\nRespond with the correct answer to the following question in a complete sentence. Question: {ques}"
    return question

def get_answer(data, model, image_processor, tokenizer, max_len, device):
    ques = data['Q']
    images = []
    titles = []
    captions = []
    for img in data['img_posFacts']:
        images.append(read_image(img['image_id']))
        titles.append(img['title'])
        captions.append(img['caption'])
    if len(images) == 2:
        prompt = get_multihop_prompt(ques, titles, captions)
    else:
        prompt = get_si_prompt(ques, titles, captions)
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

def read_image(image_id):
    with open(images_dir, "r") as fp:
        fp.seek(lineidx[int(image_id)%10000000])
        imgid, img_base64 = fp.readline().strip().split('\t')
    assert int(image_id) == int(imgid), f'{image_id} {imgid}'
    im = Image.open(BytesIO(base64.b64decode(img_base64)))
    return im

## Load model
# pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
def main():
    out_file = "/mnt/disks/data/webqa/ans_val_imgs_base.txt"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, load_8bit=True, **llava_model_args)
    model.eval()
    f = open(out_file, 'a')
    count = 0
    for id,d in tqdm(val_data.items()):
        count += 1
        ans = get_answer(d, model, image_processor, tokenizer, max_length, device)
        print(f"Ques: {d['Q']}\nAssistant: {ans}\nGT: {d['A'][0]}\n")
        f.write(f"{id}\t{ans}\n")
        if count == 5:
            break
    f.close()
if __name__ == '__main__':
    main()