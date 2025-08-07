import requests
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

def getbearertoken(api_key):
    api_key = os.getenv("WX_API_KEY") # "mUH7s1yeD8TG2ZoT7mWzIshFnxJ6y4OeNoHV-jSDOqBp"

    iam_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }

    data = {
        "apikey": api_key,
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
    }

    response = requests.post(iam_url, headers=headers, data=data)

    if response.status_code == 200:
        token_response = response.json()
        access_token = token_response["access_token"]
        return access_token
    else:
        print(f" Failed to get token: {response.status_code}")
        print(response.text)
