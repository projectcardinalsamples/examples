#This is a private copy of https://github.com/projectcardinalsamples/examples/blob/main/huggingface_pytorch_examples/hate_speech_detection.py
import requests
from transformers import pipeline
from detoxify import Detoxify

#GPT3 based model for text generative AI
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')

#Generate some text
result = generator("who is elon musk", max_length=100, do_sample=True, temperature=0.9)

quit(0)
