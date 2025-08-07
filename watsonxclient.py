import requests
import json
from typing import Dict, Any, Optional
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from requests.auth import HTTPBasicAuth
import os
from dotenv import load_dotenv
from instanceroute import getbearertoken




class WatsonXClient:
    """Client for interacting with Groq LLM API"""

    def __init__(self):
        # load env variables
        load_dotenv()
        
        # Set env variables
        self.API_KEY = os.getenv("WX_API_KEY")
        self.BASE_URL = os.getenv("WX_BASE_URL")
        self.PROJECT_ID = os.getenv("WX_PROJECT_ID") 
        self.MODEL_ID = os.getenv("WX_MODEL_ID")
        self.bearertoken = getbearertoken(self.API_KEY)
        self.headers = {
        "Authorization": f"bearer {self.bearertoken}",
        "Content-Type": "application/json"
        }
        
        


    def generate_completion(
        self, 
        prompt: str, 
        model: Optional[str] = os.getenv("WX_MODEL_ID"),
        max_tokens: int = 4000,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
        max_retries: int = 3
    ) -> Optional[str]:
        """
        Generate completion using WatsonX API with error handling and rate limit retry

        Args:
            prompt: The user prompt
            model: Model to use (defaults to default_model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            max_retries: Maximum number of retries for rate limits

        Returns:
            Generated text or None if failed
        """
        if not self.API_KEY:
            print("Error: WatsonX not found in environment variables")
            return None

        import time

        for attempt in range(max_retries + 1):
            try:
                messages = []

                if system_prompt:
                    messages.append({
                        "role": "system",
                        "content": system_prompt
                    })

                messages.append({
                    "role": "user", 
                    "content": prompt
                })

                # payload = {
                #     "model": model or self.default_model,
                #     "messages": messages,
                #     "max_tokens": max_tokens,
                #     "temperature": temperature,
                #     "stream": False
                # }

                # response = requests.post(
                #     f"{self.base_url}/chat/completions",
                #     headers=self.headers,
                #     json=payload,
                #     timeout=120  # 2 minute timeout for large requests
                # )

                wx_payload = {
                "model_id": self.MODEL_ID,
                "project_id": self.PROJECT_ID,
                "messages": messages,
                "parameters": {
                    "max_tokens": 200,
                    "time_limit": 1000
                        }
                    }
                response = requests.post(
                f"{self.BASE_URL}/ml/v1/text/chat?version=2024-05-01",
                headers=self.headers,
                data=json.dumps(wx_payload)
                )

                if response.status_code == 200:
                    data = response.json()
                    print(data)
                    return data["choices"][0]["message"]["content"]
                else:
                    print(f"watsonX API error: {response.status_code} - {response.text}")
                    if response.status_code == 429:  # Rate limit error
                        error_str = response.text
                        if attempt < max_retries:
                            import re
                            wait_match = re.search(r'try again in (\d+\.?\d*)s', error_str)
                            if wait_match:
                                wait_time = float(wait_match.group(1)) + 1
                            else:
                                wait_time = (attempt + 1) * 5
                            print(f"Rate limit hit, waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                            time.sleep(wait_time)
                            continue
                        else:
                            print("Max retries exceeded due to rate limits")
                            return None
                    else:
                        return None


            except requests.exceptions.Timeout:
                print("watsonX API request timed out")
                return None
            except requests.exceptions.RequestException as e:
                print(f"watsonX API request failed: {str(e)}")
                return None
            except Exception as e:
                print(f"Unexpected error in WatsonX client: {str(e)}")
                return None

        return None

if __name__ == "__main__":
    client = WatsonXClient()
    prompt = "Summarize the latest trends in artificial intelligence."
    system_prompt = "You are an expert AI assistant."
    result = client.generate_completion(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=500,
        temperature=0.2
    )
    # print("WatsonX Completion:", result)


