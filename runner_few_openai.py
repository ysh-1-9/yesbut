########
import base64
import requests
api_key = ""
headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

def generate(image_bytes, prompt, verbose = False):
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
            "role": "system",
            "content": [{"type": "text", "text": prompt}]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                  "detail": "low"
              }
            }
          ]
        }
      ],
      "max_tokens": 256,
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    if verbose:
        from PIL import Image
        image = Image.open(image_path)
        display(image)
        print(response)
    if "usage" not in response and "error" in response:
        raise Exception(response["error"]["message"])
    return response["usage"], response["choices"][0]["message"]["content"]

    
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
import os, json
import io
from tqdm import tqdm

## schema [{"image_path": <>, "prompt": <>, "usage": {"prompt_tokens": ...}}]
with open("gpt4-usages.json", "r") as f:
    usages = json.load(f)
    total_usage = sum(x["usage"]["total_tokens"] for x in usages)

outpath = "outputs/annotations/first_round/fewshot/gpt4-vision-fewshot.json"
inpath = "images"
try:
    with open(outpath, "r") as f:
        outputs = json.load(f)
except FileNotFoundError:
    print("starting from zero")
    outputs = {}

with open("fewshot.json", "r") as f:
    files = json.load(f)

current_usage = 0
redo_files = []
pbar = tqdm(files.items())
for filename, examples in pbar:
    if filename in outputs and outputs[filename]:
        continue
    try:
        image = create_example(filename, [e[0] for e in examples])
        #display(image)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")  # You can change the format to PNG or others if needed
        image_bytes = buffered.getvalue()
        prompt = f"The given image has 3 sub images, 2 examples and one question. The first image is funny/satirical because {examples[0][1]}. The second image is funny/satirical because {examples[1][1]}. Why is the third image funny/satirical?"
        usage, output = generate(image_bytes, prompt)
    except Exception as e:
        print("Caught exception: ", str(e))
        print("Could not do: ",filename)
        usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        output = ""

    outputs[filename] = output
    with open(outpath, "w") as f:
        json.dump(outputs, f, indent=4)

    usages.append({"image_path":filename, "prompt": prompt, "usage": usage})
    with open("gpt4-usages.json", "w") as f:
        json.dump(usages, f, indent=2)
    
    current_usage+=usage["total_tokens"]
    total_usage+=usage["total_tokens"]
    pbar.set_postfix({"current_usage": current_usage, "total_usage": total_usage})

with open(outpath, "w") as f:
    json.dump(outputs, f, indent=4)

print("Completed")
