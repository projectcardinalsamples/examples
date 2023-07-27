#This is a private copy of https://github.com/projectcardinalsamples/examples/blob/main/huggingface_pytorch_examples/hate_speech_detection.py
from detoxify import Detoxify
import requests
import sys
from transformers import pipeline

#GPT3 based model for text generative AI
generator = pipeline('text-generation', model=sys.argv[1])

#Read the sentences and then for each: 1) generate text and, 2) assess toxicity of the generated text
with open(sys.argv[2]) as file:
    while line := file.readline():
        print(line.rstrip())

        result = generator(line.rstrip(), max_length=100, do_sample=True, temperature=0.9)


