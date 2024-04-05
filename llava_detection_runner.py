import sys

# Add your custom path
custom_path = "/home/abhilash_pg/recipe_llava/LLaVA"
sys.path.append(custom_path)

from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
import torch

model_path = "4bit/llava-v1.5-13b-3GB"
kwargs = {"device_map": "auto"}
kwargs['load_in_4bit'] = True
kwargs['quantization_config'] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device='cuda')
image_processor = vision_tower.image_processor

##############################################################
from PIL import Image
import os
import requests
from io import BytesIO
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import TextStreamer

def caption_image(image, prompt, verbose=False, cot=False):
    image = image.convert('RGB')
    disable_torch_init()
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    inp = f"{roles[0]}: {prompt}"
    labels = ["Y", "N"]
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()
    if cot:
        raw_prompt+="Lets think step by step. This image has 2 halves - they depict "
        max_new_tokens = 1024
    else:
        raw_prompt+=" ["
        max_new_tokens = 8
        
    if verbose:
        print("PROMPT: ",raw_prompt)
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
      output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.7,
                                  max_new_tokens=max_new_tokens, use_cache=True, stopping_criteria=[stopping_criteria])
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    output0 = outputs.rsplit('</s>', 1)[0]

    input_ids.to("cpu")
    del input_ids
    output_ids.to("cpu")
    del output_ids

    output = None
    if cot:
        if verbose:
            print(output0)
        raw_prompt+=output0+". Hence the correct option out of Y/N is ["

        input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
          output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2,
                                      max_new_tokens=8, use_cache=True, stopping_criteria=[stopping_criteria])
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs
        output = outputs.rsplit('</s>', 1)[0]

        input_ids.to("cpu")
        del input_ids
        output_ids.to("cpu")
        del output_ids

    image_tensor.to("cpu")
    del image_tensor
    
    return image, output0, output

prompt = '''You are an AI expert in detecting humour or satire. User gives you an image, and you have to make a choice "Y" or "N".
###Instructions: The image contains 2 halves - left (called yes) and right (called but). The images in both might be related and have a funny overall meaning, or they may make no sense. Make a choice "Y" if the combined image is funny or "N" if its not funny:
###Output format: one character, only one of [Y] or [N]'''
image, _, output = caption_image(Image.open("images/20240101_172315.jpg"), prompt, True, True)
print(output)
#display(image.resize((256,256)))

##############################################################

import os,json
from tqdm import tqdm
#from validate import validate_file
import random

outpath = "outputs/detection/llava.json"
inpaths = ["images","yesbut_second_round","yesbut_second_round_negatives", "yesbut_third_round", "yesbut_third_round_negatives"]

try:
    with open(outpath, "r") as f:
        outputs = json.load(f)
except FileNotFoundError:
    print("starting from zero")
    outputs = {}
    
    
total, correct, adhering = 0,0,0
files = sum(([os.path.join(folder, file) for file in os.listdir(folder) if "ipynb" not in file] for folder in inpaths),[])   
random.shuffle(files)
pbar = tqdm(files)
for filepath in pbar:
    folder, file = filepath.split("/")
    if filepath in outputs and outputs[filepath] and outputs[filepath].get("pred",None):
        total+=1
        correct+= 1 if ((outputs[filepath]["pred"][0]=="Y" and "negative" not in folder) or (outputs[filepath]["pred"][0]=="N" and "negative" in folder)) else 0
        adhering+= 1 if outputs[filepath]["pred"][0] in ["Y", "N"] else 0
        pbar.set_postfix({"folder": folder, "total": total, "accuracy": correct/total, "adherance": adhering/total})
        continue
    image = Image.open(filepath)
    _, reason, pred = caption_image(image, prompt, verbose=False, cot=True)
    output = {"reason":reason, "pred": pred}
    
    outputs[filepath]= output
    with open(outpath, "w") as f:
        json.dump(outputs, f, indent=4)

    total+=1
    correct+= 1 if ((output["pred"][0]=="Y" and "negative" not in folder) or (output["pred"][0]=="N" and "negative" in folder)) else 0
    adhering+= 1 if output["pred"][0] in ["Y", "N"] else 0
    pbar.set_postfix({"folder": folder, "total": total, "accuracy": correct/total, "adherance": adhering/total})
with open(outpath, "w") as f:
    json.dump(outputs, f, indent=4)
# if not validate_file(outpath):
#     print("Validation failed!")                           
