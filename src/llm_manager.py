import os
import json

# Import API clients
from litellm import completion

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")


class UsageTracker:
    def __init__(self):
        self.input_tokens = 0

        self.completion_tokens = 0
        self.thinking_tokens = 0
        self.output_tokens = 0

        self.total_tokens = 0
    
    def get_usage(self):
        return {"input_tokens": self.input_tokens, "completion_tokens": self.completion_tokens, "thinking_tokens": self.thinking_tokens, "output_tokens": self.output_tokens, "total_tokens": self.total_tokens}

    def update_usage(self, usage, server_type=None):
        if server_type == "litellm":
            self.input_tokens += usage["prompt_tokens"]
            self.output_tokens += usage["completion_tokens"]
            self.total_tokens += usage["total_tokens"]
            self.thinking_tokens += usage["thinking_tokens"] 
        else:
            self.input_tokens += usage["prompt_tokens"]
            self.output_tokens += usage["completion_tokens"]
            self.total_tokens += usage["total_tokens"]
            self.thinking_tokens += usage["thinking_tokens"]

class KiwiLLM:
    def __init__(self, model_name: str, server_type: str, server_address: str, server_port: int, server_config: dict, save_path: str):
        self.model_name = model_name
        self.server_type = server_type
        self.server_address = server_address
        self.server_port = server_port
        self.server_config = server_config
        self._setup_client()

        # Usage tracking
        self.save_path = save_path
        self.total_usage = UsageTracker()
        self.open_ckpts = {}
        self.closed_ckpts = {}

        self.load_state()

    def open_ckpt(self, ckpt_name: str):
        self.open_ckpts[ckpt_name] = UsageTracker()
    
    def close_ckpt(self, ckpt_name: str):
        self.closed_ckpts[ckpt_name] = self.open_ckpts[ckpt_name]
        del self.open_ckpts[ckpt_name]
    
    def save_state(self):
        data = {
            "total_usage": self.total_usage.get_usage(),
            "open_ckpts": {ckpt_name: ckpt_usage.get_usage() for ckpt_name, ckpt_usage in self.open_ckpts.items()},
            "closed_ckpts": {ckpt_name: ckpt_usage.get_usage() for ckpt_name, ckpt_usage in self.closed_ckpts.items()}
        }
        with open(os.path.join(self.save_path, "usage_state.json"), "w") as f:
            json.dump(data, f)

    def load_state(self):
        if not os.path.exists(os.path.join(self.save_path, "usage_state.json")):
            return
        with open(os.path.join(self.save_path, "usage_state.json"), "r") as f:
            data = json.load(f)
        self.total_usage = UsageTracker()
        self.open_ckpts = {}
        self.closed_ckpts = {}
        self.total_usage.update_usage(data["total_usage"])
        for ckpt_name, ckpt_usage in data["open_ckpts"].items():
            self.open_ckpts[ckpt_name] = UsageTracker()
            self.open_ckpts[ckpt_name].update_usage(ckpt_usage)
        for ckpt_name, ckpt_usage in data["closed_ckpts"].items():
            self.closed_ckpts[ckpt_name] = UsageTracker()
            self.closed_ckpts[ckpt_name].update_usage(ckpt_usage)
    
    def _setup_client(self):
        if self.server_type == "google":
            self.client = genai.Client(api_key=GEMINI_KEY)
        elif self.server_type == "openai":
            self.client = OpenAI(api_key=OPENAI_KEY)
        elif self.server_type == "anthropic":
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        elif self.server_type == "deepseek":
            self.client = OpenAI(api_key=DEEPSEEK_KEY)
        else:
            raise NotImplementedError


    def call_llm(self, prompt: str, system_prompt: str = "You are a helpful assistant", temperature: float = 0.0, top_p: float = 1.0, top_k: int = 50, max_tokens: int = 128, num_completions: int = 1):


        pass


