import os
import openai
import numpy as np
from typing import Optional, Union, List, Any
import time
import base64

# from .. import LanguageModel, GenerateOutput
from openai import OpenAI

class GenerateOutput:
    def __init__(self, text: List[str], log_prob=None):
        self.text = text
        self.log_prob = log_prob

class OpenAIMMModel:
    def __init__(self, 
                 model: str, 
                 max_tokens: int = 2048, 
                 temperature: float = 0.0, 
                 additional_prompt: Optional[str] = None):
        """
        Initialize the OpenAI Multimodal Model wrapper.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Safely retrieve API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.client = OpenAI(api_key=api_key)
        self.additional_prompt = additional_prompt

    def _encode_image(self, image_path: Union[str, bytes]) -> str:
        """
        Encode an image to base64 for API transmission.
        """
        if isinstance(image_path, str):
            # If it's a file path, read the file
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        elif isinstance(image_path, bytes):
            # If it's already bytes, encode directly
            return base64.b64encode(image_path).decode('utf-8')
        else:
            raise ValueError("Image must be a file path or bytes")

    def generate(self,
                 prompt: Optional[Union[str, List[str]]],
                 image: Optional[Union[str, bytes]] = None,
                 max_tokens: Optional[int] = None,
                 top_p: float = 1.0,
                 num_return_sequences: int = 1,
                 rate_limit_per_min: Optional[int] = 20,
                 stop: Optional[Union[str, List[str]]] = None,
                 logprobs: Optional[int] = None,
                 temperature: Optional[float] = None,
                 retry: int = 64,
                 **kwargs) -> GenerateOutput:
        """
        Generate text response using OpenAI's API.
        """
        # Normalize prompt to a single string
        if isinstance(prompt, list):
            prompt = prompt[0]
        
        # Use default or provided parameters
        gpt_temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        logprobs = logprobs or 0

        # Ensure stop is either None, a string, or a list of strings
        if stop is not None and not isinstance(stop, (str, list)):
            stop = None

        # Retry mechanism for API calls
        for i in range(1, retry + 1):
            try:
                # Optional rate limiting
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)

                # Prepare messages
                messages = []
                
                # Handle multimodal input
                if image:
                    # Encode image to base64
                    encoded_image = self._encode_image(image)
                    
                    messages.append({
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{encoded_image}"}}
                        ]
                    })
                else:
                    # Text-only input
                    messages.append({"role": "user", "content": prompt})
                
                # Prepare generation parameters
                generation_params: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": gpt_temperature,
                    "top_p": top_p,
                    "n": num_return_sequences,
                }
                
                # Add stop parameter only if it's not None
                if stop is not None:
                    generation_params["stop"] = stop

                # Generate response
                response = self.client.chat.completions.create(**generation_params)

                return GenerateOutput(
                    text=[choice.message.content for choice in response.choices],
                    log_prob=None
                )

            except Exception as e:
                print(f"Error occurred: {e}, retry attempt {i}")
                time.sleep(i)

        raise RuntimeError("Failed to generate output after maximum retry attempts")

    def extract_image_features(self, image: Union[str, bytes], **kwargs) -> np.ndarray:
        """
        Extract features from an image.
        """
        try:
            # Encode the image
            encoded_image = self._encode_image(image)
            
            # Use chat completions for image understanding
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{encoded_image}"}}
                    ]
                }]
            )
            
            # Return a placeholder feature vector
            return np.array([float(len(response.choices[0].message.content))])

        except Exception as e:
            print(f"Error extracting image features: {e}")
            raise

if __name__ == '__main__':
    # Example usage
    try:
        # Ensure to set the OPENAI_API_KEY environment variable first
        model = OpenAIMMModel(model='gpt-4o')

        # Multimodal example
        multimodal_output = model.generate(
            prompt='Describe what you see in this image.', 
            image='image.jpg'
        )
        print("Multimodal Output:", multimodal_output.text)

    except Exception as e:
        print(f"Error in example usage: {e}")
