import torch
import os
import sys
from uuid import uuid4
from torch.utils.data import IterableDataset
import datasets

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.interactions.base import BaseInteraction

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from src.run_utils import find_highest_sample_id, fetch_baseline_results, write_kernel_to_disk, write_eval_result_to_separate_file
from src.utils import extract_last_code

from main.evaluation_utils import send_evaluation_request, send_batch_evaluation_request, EvaluationWorkArgs, serialize_work_args, is_generated_kernel_used


# RUNS_DIR = "/data/user_data/gyeongwk/KernelBench/grpo/runs"
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
RUN_NAME = "grpo_train_Qwen2.5-7B-Instruct-SFT"
EVAL_SERVER_HOST = "babel-11-9"
EVAL_SERVER_PORT = 8083
NUM_GENERATIONS = 8
HARDWARE = "A6000_babel"

os.makedirs(os.path.join(RUNS_DIR, RUN_NAME), exist_ok=True)

# Define custom reward functions
def reward_from_exec_result(level, problem, exec_result):
    if exec_result.correctness:
        try:
            baseline_results = fetch_baseline_results(level, problem, HARDWARE)
            speedup = baseline_results["mean"] / exec_result.runtime
            return 0.3 + float(speedup)
        except Exception as e:
            print(f"Error fetching baseline results for level {level} problem {problem}: {e}")
            return 0.3
    else:
        return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, i=None):
    split, level, problem = extra_info['split'], extra_info['level'], extra_info['problem']
    run_dir = os.path.join(RUNS_DIR, RUN_NAME)

    thread_id = i % NUM_GENERATIONS
    if split == "train":
        sample_id = find_highest_sample_id(run_dir, level, problem, thread_id, NUM_GENERATIONS) # batch_size
    else:
        sample_id = find_highest_sample_id(run_dir, level, problem, 0, 1) # just find the next sample_id


    response_path = os.path.join(run_dir, f"level_{level}_problem_{problem}_sample_{sample_id}_response.txt")
    with open(response_path, "w") as f:
        f.write(solution_str)

    kernel_src = extract_last_code(solution_str, ["python", "cpp"])
    kernel_name = f"level_{level}_problem_{problem}_sample_{sample_id}"

    if kernel_src is not None:
        write_kernel_to_disk(run_dir, level, problem, sample_id, kernel_src)

        work_args=EvaluationWorkArgs(level=level, problem_id=problem, sample_id=sample_id, device=torch.device("cuda"))
        exec_result = send_evaluation_request(EVAL_SERVER_HOST, EVAL_SERVER_PORT, work_args, RUN_NAME, kernel_src, kernel_name)
        write_eval_result_to_separate_file(level, problem, sample_id, exec_result, run_dir)
        
        return reward_from_exec_result(level, problem, exec_result)

    print(f"No kernel src found for level {level} problem {problem} sample {sample_id}")
    return 0.0


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    run_dir = os.path.join(RUNS_DIR, RUN_NAME)

    work_args_list = []
    job_list = []
    thread_ids = {}
    rewards = {}
    for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos):
        level = extra_info['level']
        problem = extra_info['problem']
        key = f"{level}_{problem}"
        if key not in thread_ids:
            thread_ids[key] = find_highest_sample_id(run_dir, level, problem, 0, 1)
        else:
            thread_ids[key] += 1
        sample_id = thread_ids[key]
        work_args_list.append((level, problem, sample_id))

        response_path = os.path.join(run_dir, f"level_{level}_problem_{problem}_sample_{sample_id}_response.txt")
        with open(response_path, "w") as f:
            f.write(solution_str)

        kernel_name = f"level_{level}_problem_{problem}_sample_{sample_id}"
        kernel_src = extract_last_code(solution_str, ["python", "cpp"])

        if kernel_src is not None:
            write_kernel_to_disk(run_dir, level, problem, sample_id, kernel_src) # for debugging

            if not is_generated_kernel_used(kernel_src):
                rewards[f"{level}_{problem}_{sample_id}"] = 0.0
                continue

            work_args=EvaluationWorkArgs(level=level, problem_id=problem, sample_id=sample_id, device=torch.device("cuda"))
            job_list.append({
                "work_args": serialize_work_args(work_args),
                "run_name": RUN_NAME,
                "kernel_src": kernel_src,
                "kernel_name": kernel_name
            })
        else:
            rewards[f"{level}_{problem}_{sample_id}"] = 0.0
    
    results = send_batch_evaluation_request(EVAL_SERVER_HOST, EVAL_SERVER_PORT, job_list)

    for result, job in zip(results, job_list):
        level = job['work_args']["level"]
        problem = job['work_args']["problem_id"]
        sample_id = job['work_args']["sample_id"]
        write_eval_result_to_separate_file(level, problem, sample_id, result, run_dir)
        reward = reward_from_exec_result(level, problem, result)
        rewards[f"{level}_{problem}_{sample_id}"] = reward
    
    # Turn rewards into list in the same order as input
    rewards_list = [rewards[f"{level}_{problem}_{sample_id}"] for level, problem, sample_id in work_args_list]
    return rewards_list


# Define Interaction Environment for multi-turn support
class KernelBenchInteraction(BaseInteraction):
    def __init__(self, config):
        super().__init__(config)
        self._instance_dict = {}
    
    async def start_interaction(self, instance_id=None, ground_truth=None, **kwargs):
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "ground_truth": ground_truth,
            "response": None,
            "reward": 0.0,
        }
        return instance_id

    async def generate_response(self, instance_id, messages, **kwargs):
        content = ""
        for item in reversed(messages):
            if item.get("role") == "user":
                content = item.get("content", "")
                break
        
        self._instance_dict[instance_id]["response"] = content
        
        reward, message = await self.run_kernel_evaluation(instance_id)
        return False, message, reward, {}
    
    async def run_kernel_evaluation(self, instance_id):
        return compute_score("KernelBench", self._instance_dict[instance_id]["response"], self._instance_dict[instance_id]["ground_truth"], self._instance_dict[instance_id]["extra_info"])

    async def finalize_interaction(self, instance_id, **kwargs):
        del self._instance_dict[instance_id]


class CurriculumDataset(IterableDataset):
    def __init__(self, data_files, tokenizer, processor, config):
        # Store the original RLHFDataset for reference
        self.rlhf_dataset = RLHFDataset(data_files, tokenizer, processor, config)
        
        # Curriculum learning parameters
        self.epochs_per_stage = config.get("epochs_per_stage", 5)
        self.current_epoch = 0
        self.stage = 0
        self.num_stages = len(data_files) if isinstance(data_files, list) else 1
        
        # Store all dataframes for different stages
        self.dataframes = []
        self._load_all_stages(data_files, tokenizer, processor, config)
        
        # Current stage dataframe
        self.current_dataframe = self.dataframes[self.stage]
        self.current_indices = list(range(len(self.current_dataframe)))
        
        # Shuffle indices for current stage
        import random
        random.shuffle(self.current_indices)
        self.current_idx = 0
    
    def _load_all_stages(self, data_files, tokenizer, processor, config):
        """Load all stages of data into separate dataframes"""
        if not isinstance(data_files, list):
            data_files = [data_files]
        
        for parquet_file in data_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            
            # Apply filtering if needed (same logic as RLHFDataset)
            if config.get("filter_overlong_prompts", True):
                dataframe = self._filter_overlong_prompts(dataframe, tokenizer, processor, config)
            
            self.dataframes.append(dataframe)
    
    def _filter_overlong_prompts(self, dataframe, tokenizer, processor, config):
        """Filter prompts that are too long (same logic as RLHFDataset)"""
        max_prompt_length = config.get("max_prompt_length", 1024)
        prompt_key = config.get("prompt_key", "prompt")
        image_key = config.get("image_key", "images")
        video_key = config.get("video_key", "videos")
        num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        
        if processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            def doc2len(doc) -> int:
                messages = self._build_messages(doc)
                raw_prompt = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                images = (
                    [process_image(image) for image in messages.pop(image_key)] if image_key in messages else None
                )
                videos = (
                    [process_video(video) for video in messages.pop(video_key)] if video_key in messages else None
                )

                return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])
        else:
            def doc2len(doc) -> int:
                return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

        return dataframe.filter(
            lambda doc: doc2len(doc) <= max_prompt_length,
            num_proc=num_workers,
            desc=f"Filtering prompts longer than {max_prompt_length} tokens",
        )
    
    def _build_messages(self, example: dict):
        """Build messages from example (same logic as RLHFDataset)"""
        prompt_key = self.rlhf_dataset.prompt_key
        messages: list = example.pop(prompt_key)
        
        # Handle multimodal content if needed
        image_key = self.rlhf_dataset.image_key
        video_key = self.rlhf_dataset.video_key
        
        if image_key in example or video_key in example:
            import re
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages
    
    def advance_epoch(self):
        """Advance to the next epoch and potentially next stage"""
        self.current_epoch += 1
        
        if (self.current_epoch % self.epochs_per_stage == 0 and 
            self.stage < self.num_stages - 1):
            self.advance_stage()
   
    def advance_stage(self):
        if self.stage < self.num_stages - 1:
            self.stage += 1
            print(f"Advancing to curriculum stage {self.stage}")
            
            # Update current dataframe and indices
            self.current_dataframe = self.dataframes[self.stage]
            self.current_indices = list(range(len(self.current_dataframe)))
            
            # Shuffle indices for new stage
            import random
            random.shuffle(self.current_indices)
            self.current_idx = 0
        else:
            print(f"Already at the final stage {self.stage}")
    
    def set_stage(self, stage):
        """Set the curriculum to a specific stage"""
        if 0 <= stage < self.num_stages:
            self.stage = stage
            print(f"Setting curriculum to stage {self.stage}")
            
            # Update current dataframe and indices
            self.current_dataframe = self.dataframes[self.stage]
            self.current_indices = list(range(len(self.current_dataframe)))
            
            # Shuffle indices for new stage
            import random
            random.shuffle(self.current_indices)
            self.current_idx = 0
        else:
            raise ValueError(f"Stage {stage} is out of range. Available stages: 0-{self.num_stages-1}")
    
    def reset_iteration(self):
        """Reset the iteration state for the current stage"""
        self.current_indices = list(range(len(self.current_dataframe)))
        import random
        random.shuffle(self.current_indices)
        self.current_idx = 0
    
    def __iter__(self):
        """Iterate through the current stage's data"""
        # Reset current index for new iteration
        self.current_idx = 0
        
        while self.current_idx < len(self.current_indices):
            # Get the actual index in the dataframe
            df_idx = self.current_indices[self.current_idx]
            
            # Get the row from current dataframe
            row_dict = dict(self.current_dataframe[df_idx])
            
            # Process the row using RLHFDataset logic
            processed_row = self._process_row(row_dict)
            
            self.current_idx += 1
            yield processed_row
        self.advance_epoch()
    
    def _process_row(self, row_dict):
        """Process a row using the same logic as RLHFDataset.__getitem__"""
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.rlhf_dataset.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.rlhf_dataset.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.rlhf_dataset.image_key in row_dict and row_dict.get(self.rlhf_dataset.image_key, None) is not None:
                images = [process_image(image) for image in row_dict.pop(self.rlhf_dataset.image_key)]
                multi_modal_data["image"] = images

            videos = None
            if self.rlhf_dataset.video_key in row_dict and row_dict.get(self.rlhf_dataset.video_key, None) is not None:
                videos = [process_video(video) for video in row_dict.pop(self.rlhf_dataset.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.rlhf_dataset.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.rlhf_dataset.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.rlhf_dataset.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        # Postprocess data
        import verl.utils.torch_functional as verl_F
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.rlhf_dataset.max_prompt_length,
            pad_token_id=self.rlhf_dataset.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.rlhf_dataset.truncation,
        )

        # Handle position IDs
        if (self.rlhf_dataset.processor is not None and 
            "Qwen2VLImageProcessor" in self.rlhf_dataset.processor.image_processor.__class__.__name__):
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.rlhf_dataset.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]
        else:
            from verl.utils.model import compute_position_id_with_mask
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        # Handle raw prompt IDs
        raw_prompt_ids = self.rlhf_dataset.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.rlhf_dataset.max_prompt_length:
            if self.rlhf_dataset.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.rlhf_dataset.max_prompt_length :]
            elif self.rlhf_dataset.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.rlhf_dataset.max_prompt_length]
            elif self.rlhf_dataset.truncation == "middle":
                left_half = self.rlhf_dataset.max_prompt_length // 2
                right_half = self.rlhf_dataset.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.rlhf_dataset.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.rlhf_dataset.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        
        # Add additional fields
        if self.rlhf_dataset.return_raw_chat:
            row_dict["raw_prompt"] = messages

        if self.rlhf_dataset.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        # Add index and other metadata
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.rlhf_dataset.need_tools_kwargs)
        
        if need_tools_kwargs and not tools_kwargs:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        
        return row_dict
    
    def __len__(self):
        """Return length of current stage"""
        return len(self.current_dataframe)
    
    def get_stage_info(self):
        """Get information about current curriculum stage"""
        return {
            "current_stage": self.stage,
            "total_stages": self.num_stages,
            "current_epoch": self.current_epoch,
            "epochs_per_stage": self.epochs_per_stage,
            "stage_size": len(self.current_dataframe)
        }



