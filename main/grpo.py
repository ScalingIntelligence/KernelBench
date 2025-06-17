import verifiers as vf
from verifiers.envs import SingleTurnEnv, MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric

# Evaluation
def reward_func(prompt, completion, answer, **kwargs):
    return 1.0

kernel_rubric = Rubric(funcs=[reward_func], weights=[1.0])
    

class KernelSingleTurnEnv(SingleTurnEnv):
    def __init__(self, dataset):
        rubric = Rubric(funcs=[reward_func], weights=[1.0])
        system_prompt = "You are a kernel expert"
        super().__init__(dataset=dataset, system_prompt=system_prompt, rubric=rubric)
    
class KernelMultiTurnEnv(MultiTurnEnv):
    def __init__(self, dataset, max_turns):
        rubric = Rubric(funcs=[reward_func], weights=[1.0])
        system_prompt = "You are a kernel expert"
        super().__init__(dataset=dataset, system_prompt=system_prompt, rubric=rubric, max_turns=max_turns)
    
    def env_response(self, messages, state, **kwargs):
        # eval logic to parse response and run kernel
        pass

    def is_completed(self, messages, state, **kwargs):
        return state.get("completed", False) or state.get("attempts", 0) >= self.max_turns
    
    def score_rollout(self, rollout, **kwargs):
        pass
    


# train.py

model, tokenizer = vf.get_model_and_tokenizer(model_name)
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=vf.grpo_defaults(run_name="...")
)
trainer.train()