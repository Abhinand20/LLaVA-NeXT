
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

dataset_dir = '/mnt/disks/data/webqa/WebQA_test.json'
lineidx_dir = '/mnt/disks/data/webqa/imgs.lineidx'
images_dir = '/mnt/disks/data/webqa/imgs.tsv'
retrieved_dir = '/mnt/disks/data/webqa/WebQA_retrieved_test.json'

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

# Map test set images to IDs
img_url_to_id = {}
with open(dataset_dir, 'r') as f:
    test_data = json.load(f)

with open(retrieved_dir, 'r') as f:
    retrieved_data = json.load(f)

for qid,d in test_data.items():
    for img in d['img_Facts']:
        img_url_to_id[img['imgUrl']] = img['image_id']

with open(lineidx_dir, "r") as fp_lineidx:
    lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]

# Image context (1,2) + Text context (1,2) + Question
# For cases where Img context = 0; use multimodal = False config and re-run the generation
def get_multihop_image_prompt(titles):
    question = f"{DEFAULT_IMAGE_TOKEN} This is the first image.\nTitle: {titles[0]}\n\nNow, let's look at another image: {DEFAULT_IMAGE_TOKEN} Title: {titles[1]}\n"
    return question

def get_si_prompt(titles):
    question = f"{DEFAULT_IMAGE_TOKEN} Consider this image\nTitle: {titles[0]}\n"
    return question

def get_multihop_text_prompt(texts):
    question = f"Fact 1: {texts[0]}\n\nFact 2: {texts[1]}\n"
    return question

def get_single_text_prompt(texts):
    question = f"Fact: {texts[0]}\n"
    return question

def get_ques_prompt(q):
    question = f"Given all of the relevant context before, respond with the correct answer to the following question in a complete sentence. Question: {q}"
    return question


def get_test_answer(data, model, image_processor, tokenizer, max_len, device):
    ques = data['Q']
    images = []
    titles = []
    texts = []
    for img in data['img_Facts']:
        image_id = img_url_to_id[img['imgUrl']]
        images.append(read_image(image_id))
        titles.append(img['title'])
    for txt in data['txt_Facts']:
        texts.append(txt['fact'])
        
    # skip if no images
    if len(images) == 2:
        promptImg = get_multihop_image_prompt(titles)
    elif len(images) == 1:
        promptImg = get_si_prompt(titles)
    else:
        raise NotImplementedError

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
    print(prompt_question)

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
    out_file = "/mnt/disks/data/webqa/ans_test_imgs_base.txt"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, load_8bit=True, **llava_model_args)
    model.eval()
    f = open(out_file, 'a')
    for id,d in tqdm(retrieved_data.items()):
        try:
            ans = get_test_answer(d, model, image_processor, tokenizer, max_length, device)
            f.write(f"{id}\t{ans}\n")
        except Exception as e:
            print(e)
            continue
    f.close()

if __name__ == '__main__':
    main()
