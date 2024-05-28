import base64
from PIL import Image
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from vertexai.generative_models._generative_models import HarmCategory, HarmBlockThreshold, ResponseBlockedError
from pathlib import Path

gcp_project_name = "" # get this from your google cloud platform account
vertexai.init(project=gcp_project_name, location="us-central1")

def generate(image, prompt):
    model = GenerativeModel("gemini-pro-vision")
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    # Yes,But memes are funny because the Yes picture depicts a normal situation, and the But picture reveals something about the Yes picture that makes humans laugh. What is funny about the given Yes,But meme
    
    responses = model.generate_content(
        [prompt,image],
        generation_config={
            "max_output_tokens": 256,
            "temperature": 1,
            "top_p": 1,
            "top_k": 32,
            "candidate_count": 1
        },
        safety_settings=safety_settings,
        
    )
    #print(type(responses))
    #print(responses)
    #print(responses.text)
    return responses.text

    
#############
from PIL import Image, ImageOps, ImageDraw, ImageFont

def label_images(images):
    labels = ["Example 1", "Example 2", "Question"]
    new_images = []
    for text,image in zip(labels,images):
      new_height = int(image.size[1] * 1.1)
      new_image = Image.new("RGB", (image.size[0], new_height), color=(255, 255, 255))
      new_image.paste(image, (0, 0))
      draw = ImageDraw.Draw(new_image)
      font = ImageFont.truetype("Arial.ttf", 40)
      text_bbox = draw.textbbox(((new_image.size[0] - image.size[0]) // 2, image.size[1], (new_image.size[0] + image.size[0]) // 2, new_height), text, font=font)
      text_position = ((new_image.size[0] - text_bbox[2]) // 2, image.size[1] + (new_height-image.size[1]) // 2)
      draw.text(text_position, text, font=font, fill=(0, 0, 0))
      new_images.append(new_image)

    return new_images

def border(images):
    new_images = []
    for image in images:
        img_with_border = ImageOps.expand(image, border=2, fill='black')
        new_images.append(img_with_border)
    return new_images

def resize(images, target_width, target_height):
  new_images = []
  for image in images:
    resized_img = image.resize((target_width, target_height))
    new_images.append(resized_img)
  
  return new_images

def join_images(images, resize_flag=True, border_flag=True):
    if resize_flag:
        images = resize(images, target_width=340, target_height=256)
    if border_flag:
        images = border(images)
    n = len(images)
        
    cumulative_width = sum(img.width for img in images)
    total_height = max(img.height for img in images)
    padding = int(0.05 * cumulative_width/n)
    new_image = Image.new('RGB', (cumulative_width + (n-1) * padding, total_height), (255, 255, 255))
    current_width = 0
    for img in images:
        new_image.paste(img, (current_width, 0))
        current_width += img.width + padding
    return new_image

def stack_images(images):
    padding = int(max(img.height for img in images)*0.1)
    total_height = sum(img.height for img in images) + padding*(len(images)-1)

    new_image = Image.new('RGB', (max(img.width for img in images), total_height), (255, 255, 255))

    y = 0
    for img in images:
        new_image.paste(img, (0,y))
        y+=img.height + padding

    return new_image

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
def text_to_image(input_text, width=340, height=256):
    img = Image.new('RGB', (width, height), color = 'white')
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("Arial.ttf", 55)
    text_width, text_height = d.textsize(input_text, font=font)
    x = (width - text_width) / 2
    y = (height - text_height) / 2
    d.text((x,y), input_text, fill="black", font=font)
    return img

#############
def create_example(question, examples):
    image_paths = [f"images/{path}" for path in examples]+[f"images/{question}"]
    images = [Image.open(path) for path in image_paths]

    return stack_images(label_images(border(images)))

########
import os,json
from tqdm import tqdm
import random
from datetime import datetime
import time
import io

outpath = "outputs/annotations/first_round/fewshot/gemini-fewshot.json"

try:
    with open(outpath, "r") as f:
        outputs = json.load(f)
except FileNotFoundError:
    print("starting from zero")
    outputs = {}

with open("fewshot.json", "r") as f:
    files = json.load(f)

pbar = tqdm(files.items())
t_last = None
for filepath, examples in pbar:
    if filepath in outputs and outputs[filepath]:
        continue
    image = create_example(filepath, [e[0] for e in examples])
    #display(image)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")  # You can change the format to PNG or others if needed
    image_part = Part.from_data(data=buffered.getvalue(), mime_type="image/jpeg")
    prompt = f"The given image has 3 sub images, 2 examples and one question. The first image is funny/satirical because {examples[0][1]}. The second image is funny/satirical because {examples[1][1]}. Why is the third image funny/satirical?"
        
    try:
        output = generate(image_part, prompt)
    except Exception as e:
        print("Caught exception: ",str(e))
        if "generate_content_requests_per_minute_per_project_per_base_model" in str(e):
            print("sleeping for 120s")
            time.sleep(120)
            print("sleep over")
            try:
                output = generate(image_part, prompt)
            except Exception as e:
                print("Could not do file: ",filepath)
                output = ""
        else:
            print("Could not do file: ",filepath)
            output = ""
    
    outputs[filepath]= output
    with open(outpath, "w") as f:
        json.dump(outputs, f, indent=4)

with open(outpath, "w") as f:
    json.dump(outputs, f, indent=4)
print("Completed")
