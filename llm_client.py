from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

import requests

HOST = '127.0.0.1:5005'
URI = f'http://{HOST}/v1/completions'

class AlpacaLLM(LLM):
    
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if isinstance(stop, list):
            stop = stop + ["\n###","\nObservation:", "\nObservations:"]

        response = requests.post(
            URI,
            json={
                "prompt": prompt,
                "temperature": 0.3,
                'do_sample': True,
                'top_p': 0.1,
                'typical_p': 1,
                'repetition_penalty': 1.18,
                'top_k': 40,
                'min_length': 0,
                'no_repeat_ngram_size': 0,
                'penalty_alpha': 0,
                'seed': -1,
                'add_bos_token': True,
                'ban_eos_token': False,
                'skip_special_tokens': True,
                'max_tokens' : 512
                },
        )
        response.raise_for_status()
        # print(response.json()['choices'][0]['finish_reason'], res_json['usage'])
        # print()
        return response.json()['choices'][0]['text']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}
