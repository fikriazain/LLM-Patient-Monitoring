from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

import requests

HOST = '127.0.0.1:5000'
URI = f'http://{HOST}/v1/completions'

class AlpacaLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if isinstance(stop, list):
            stop = stop + ["\n###","\nObservation:",'\nObservation:']

        response = requests.post(
            URI,
            json={
                "prompt": prompt,
                "temperature": 0.0,
                "max_new_tokens": 256,
                "early_stopping": True,
                "stop": stop,
                'do_sample': True,
                'top_p': 0.1,
                'typical_p': 1,
                'repetition_penalty': 1.18,
                'top_k': 40,
                'min_length': 0,
                'no_repeat_ngram_size': 0,
                'num_beams': 1,
                'penalty_alpha': 0,
                'length_penalty': 1,
                'seed': -1,
                'add_bos_token': True,
                'truncation_length': 2048,
                'ban_eos_token': False,
                'skip_special_tokens': True,
                "max_tokens": 256,
            },
        )
        response.raise_for_status()
        return response.json()['choices'][0]['text']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}
