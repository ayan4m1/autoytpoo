import torch
import argparse
import datetime

from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration, pipeline
from TTS.api import TTS
from torch import autocast
from random import random
from os import makedirs
from math import floor
from sys import stdout
from PIL import Image
from subprocess import run

stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser(description='Generate a cursed video')

parser.add_argument('--temperature', type=float, default=0.7,
                    help='Randomness of generated text')
parser.add_argument('--width', type=int, default=768,
                    help='Output vide width in pixels')
parser.add_argument('--height', type=int, default=432,
                    help='Output vide height in pixels')
parser.add_argument('--steps', type=int, default=20,
                    help='Number of steps per image')
args = parser.parse_args()

device = torch.device('cuda')

project_name = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')


def get_blip_pipe():
    path = './models/blip-image-captioning-large'
    processor = BlipProcessor.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = BlipForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.float16)
    return pipeline("image-to-text", model=model, tokenizer=tokenizer, image_processor=processor, device=0)


def get_lexart_pipe():
    path = "./models/promptgen-lexart"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)


def get_llama_pipe():
    path = "./models/llama-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)


def get_coqui_api():
    # model="tts_models/multilingual/multi-dataset/xtts_v2"
    model = "tts_models/en/jenny/jenny"
    return TTS(model).to(device)


def get_sd_pipe():
    scheduler = EulerAncestralDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        steps_offset=1
    )
    return StableDiffusionPipeline.from_pretrained("./models/stable-diffusion-v1-5", scheduler=scheduler).to(device)


def get_project_path(filename):
    makedirs(f'./output/{project_name}', exist_ok=True)
    return f'./output/{project_name}/{filename}'


base_prompt = 'Input: Generate a text description of an image that would not make sense. It should be surreal and use simple language.\n\nOutput: '

print('1/6 :: Generating base prompt...')

llama_pipe = get_llama_pipe()
llama_output = llama_pipe(base_prompt, eos_token_id=llama_pipe.tokenizer.eos_token_id, do_sample=True, temperature=args.temperature, max_new_tokens=50)[
    0]['generated_text'].replace(base_prompt, '').strip()

print(f'Base prompt: {llama_output}')

base_script_prompt = f'Input: Generate a voiceover script in the style of a drunken David Attenborough describing the following scene: "{llama_output}". Answer with the script directly, do not prepend any text.\n\nOutput: '

print('2/6 :: Generating voiceover script...')

llama_script_output = llama_pipe(base_script_prompt, eos_token_id=llama_pipe.tokenizer.eos_token_id,
                                 do_sample=True, temperature=args.temperature, max_new_tokens=300)[0]['generated_text'].replace(base_script_prompt, '').strip()

with open(get_project_path('script.txt'), 'w') as script_file:
    script_file.write(llama_script_output)

del llama_pipe
torch.cuda.empty_cache()

print('3/6 :: Turning base prompt into SD prompt...')

lexart_pipe = get_lexart_pipe()
lexart_output = lexart_pipe(llama_output, max_new_tokens=30)[
    0]['generated_text'].strip()

print(f'SD prompt: {lexart_output}')

del lexart_pipe
torch.cuda.empty_cache()

image_path = get_project_path('bg.png')

with autocast("cuda"):
    with torch.inference_mode():
        sd_pipe = get_sd_pipe()

        print('4/5 :: Generating SD image...')

        image = sd_pipe(prompt=lexart_output, num_inference_steps=args.steps,
                        generator=torch.Generator('cuda').manual_seed(
                            floor(random() * 1000000000)),
                        height=args.height, width=args.width).images[0]
        image.save(image_path)

        del sd_pipe
        torch.cuda.empty_cache()

print('5/6 :: Generating voiceover audio...')

audio_path = get_project_path('voiceover.wav')

coqui_api = get_coqui_api()
coqui_api.tts_to_file(text=llama_script_output, file_path=audio_path)

del coqui_api
torch.cuda.empty_cache()

print('5/6 :: Extracting title from SD image...')

blip_pipe = get_blip_pipe()
blip_output = blip_pipe(Image.open(image_path).convert('RGB'))[
    0]['generated_text']

with open(get_project_path('title.txt'), 'w') as title_file:
    title_file.write(blip_output)

del blip_pipe
torch.cuda.empty_cache()

run(['ffmpeg', '-loop', '1', '-i', image_path, '-i', audio_path,
    '-shortest', get_project_path('final.mp4')])
