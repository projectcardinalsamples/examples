#This is a private copy of https://github.com/projectcardinalsamples/examples/blob/main/huggingface_pytorch_examples/hate_speech_detection.py
from detoxify import Detoxify
import requests
import sys
from transformers import pipeline

#GPT3 based model for text generative AI
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')

