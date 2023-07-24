import requests
from transformers import pipeline

classifier=pipeline('sentiment-analysis')
result=classifier('We are very happy to show you the 🤗 VMware Application Catalog library.')
quit(round(result[0]['score'] - 1)) # Returns 0 on POSITIVE sentiment
