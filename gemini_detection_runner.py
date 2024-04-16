import base64
from PIL import Image
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from vertexai.generative_models._generative_models import HarmCategory, HarmBlockThreshold, ResponseBlockedError
from pathlib import Path

gcp_project_name = "" # get this from your google cloud platform account
vertexai.init(project=gcp_project_name, location="us-central1")

prompt = '''You are an AI expert in detecting humour or satire. User gives you an image, and you have to make a choice "Y" or "N".
###Instructions: Users image has 2 halves called yes and but, and the combination of those might make no sense at all, or be extremely funny. Your job is to find out which one it is and output Y if its EXTREMELY funny and N for otherwise.
###Output format: one character, exactly either "Y" or "N"'''
def generate(image):
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
            "max_output_tokens": 8,
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


with open("yesbut_third_round_negatives/20240101_172858_3D_YES_STICK_BUT.jpg", "rb") as f:
    data = f.read()
    image1 = Part.from_data(data=data, mime_type="image/jpeg")
    #print(image1)

generate(image1)



###################################

import os,json
from tqdm import tqdm
import random

outpath = "outputs/detection/gemini.json"
inpaths = ["images","yesbut_second_round","yesbut_second_round_negatives", "yesbut_third_round", "yesbut_third_round_negatives"]

try:
    with open(outpath, "r") as f:
        outputs = json.load(f)
except FileNotFoundError:
    print("starting from zero")
    outputs = {}

total, correct, adhering, y = 0,0,0,0
files = sum(([os.path.join(folder, file) for file in os.listdir(folder) if file[-3:]=="jpg"] for folder in inpaths),[])   
random.shuffle(files)
pbar = tqdm(files)
for filepath in pbar:
    folder,file = filepath.split('/')
    if filepath in outputs and outputs[filepath]:
        total+=1
        pred = outputs[filepath].strip()[0]
        correct += 1 if ((pred=="Y" and "negative" not in folder) or (pred=="N" and "negative" in folder)) else 0
        adhering += 1 if pred in ["Y", "N"] else 0
        y += 1 if pred=="Y" else 0
        pbar.set_postfix({"folder": folder, "total": total, "accuracy": correct/total, "adherance": adhering/total, "y%": y/adhering if adhering>0 else 0})
        continue
    with open(filepath, "rb") as f:
        data = f.read()
        image = Part.from_data(data=data, mime_type="image/jpeg")
    #display(Image.open(os.path.join("images",filename)).convert('RGB'))
    #print(filename)
    try:
        output = generate(image)
    except Exception as e:
        print("Caught exception: ",str(e))
        print("Could not do file: ",filename)
        output = ""
    
    outputs[filepath]= output
    with open(outpath, "w") as f:
        json.dump(outputs, f, indent=4)
        
    total+=1
    correct += 1 if outputs[filepath] and ((outputs[filepath].strip()[0]=="Y" and "negative" not in folder) or (outputs[filepath].strip()[0]=="N" and "negative" in folder)) else 0
    adhering += 1 if outputs[filepath] and outputs[filepath].strip()[0] in ["Y", "N"] else 0
    y += 1 if outputs[filepath] and outputs[filepath].strip()[0] == "Y" else 0
    pbar.set_postfix({"folder": folder, "total": total, "accuracy": correct/total, "adherance": adhering/total, "y%": y/adhering if adhering>0 else 0})

with open(outpath, "w") as f:
    json.dump(outputs, f, indent=4)