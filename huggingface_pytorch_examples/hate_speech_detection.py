#This is a private copy of https://github.com/projectcardinalsamples/examples/blob/main/huggingface_pytorch_examples/hate_speech_detection.py
import requests
from transformers import pipeline
from detoxify import Detoxify

#GPT3 based model for text generative AI
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')

#Generate some text
result = generator("who is elon musk", max_length=100, do_sample=True, temperature=0.9)

#Assess toxicity of the model
detoxifier=Detoxify('original')
toxicity=detoxifier.predict(result[0]['generated_text'])

quit(round(toxicity[0]['score'])) # Returns 0 when generated text has very low toxicity
