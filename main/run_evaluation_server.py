#!/usr/bin/env python3
"""
Script to start the evaluation server.
Usage: python run_evaluation_server.py --port <port> 
"""

import os
import socket
import threading
import pickle
from typing import Optional
import logging
import torch
import multiprocessing as mp
from tqdm import tqdm
import time

from configs import parse_eval_server_args, RUNS_DIR
from evaluation_utils import evaluate_single_sample_in_separate_process, KernelExecResult, deserialize_work_args
from src.utils import set_gpu_arch


"""
Evaluation server code that handles client requests across multiple GPUs.
Each GPU is used for a single kernel evaluation at a time.
"""



class GPUDeviceManager:
    """Thread-safe GPU device manager for the evaluation server"""
    
    def __init__(self, num_gpus: int):
        self.num_gpus = num_gpus
        self.available_gpus = set(range(num_gpus))
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
    
    def acquire_gpu(self) -> Optional[int]:
        """Acquire a free GPU device. Returns None if no GPU is available."""
        with self.lock:
            if self.available_gpus:
                gpu_id = self.available_gpus.pop()
                return gpu_id
            return None
    
    def wait_for_gpu(self) -> int:
        """Wait until a GPU becomes available and then acquire it."""
        with self.condition:
            while not self.available_gpus:
                logging.info("No GPU available, waiting...")
                self.condition.wait()
            gpu_id = self.available_gpus.pop()
            logging.info(f"Acquired GPU {gpu_id} after waiting")
            return gpu_id
    
    def release_gpu(self, gpu_id: int):
        """Release a GPU device back to the pool."""
        with self.condition:
            if 0 <= gpu_id < self.num_gpus:
                self.available_gpus.add(gpu_id)
                self.condition.notify()  # Notify waiting threads
    
    def get_available_count(self) -> int:
        """Get the number of available GPUs."""
        with self.lock:
            return len(self.available_gpus)
    
    def get_used_count(self) -> int:
        """Get the number of GPUs currently in use."""
        with self.lock:
            return self.num_gpus - len(self.available_gpus)


def get_gpu_status_info() -> dict:
    """Get information about available GPUs"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    num_gpus = torch.cuda.device_count()
    gpu_info = {
        "total_gpus": num_gpus,
        "gpu_details": []
    }
    
    for i in range(num_gpus):
        try:
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory
            gpu_info["gpu_details"].append({
                "id": i,
                "name": gpu_name,
                "total_memory_gb": gpu_memory / (1024**3)
            })
        except Exception as e:
            gpu_info["gpu_details"].append({
                "id": i,
                "error": str(e)
            })
    
    return gpu_info


def start_evaluation_server(port: int, configs):
    """
    Start a server that listens for evaluation requests on the specified port.
    
    Args:
        port: Port number to listen on
        configs: Configuration object
    
    The server automatically detects and handles both single and batch requests:
    
    Single request structure:
    {
        'work_args': EvaluationWorkArgs (serialized, device will be ignored),
        'kernel_src': str,
        'kernel_name': str,
        'run_name': str
    }
    
    Batch request structure:
    {
        'batch': [
            {
                'work_args': EvaluationWorkArgs (serialized, device will be ignored),
                'kernel_src': str,
                'kernel_name': str,
                'run_name': str
            },
            ...
        ]
    }
    
    Returns evaluation results as KernelExecResult objects (single) or list of KernelExecResult objects (batch).
    """
    # Initialize GPU device manager
    num_gpus = torch.cuda.device_count()
    gpu_manager = GPUDeviceManager(num_gpus)
    logging.info(f"Initialized GPU manager with {num_gpus} GPUs")
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind(('0.0.0.0', port))
        server_socket.listen(5)
        logging.info(f"Evaluation server started on port {port}")
        
        while True:
            client_socket, address = server_socket.accept()
            logging.info(f"Connection from {address}")
            
            # Handle each client in a separate thread
            client_thread = threading.Thread(
                target=handle_client,
                args=(client_socket, configs, gpu_manager)
            )
            client_thread.start()
            
    except KeyboardInterrupt:
        logging.info("Server shutting down...")
    except Exception as e:
        logging.info(f"Server error: {e}")
    finally:
        server_socket.close()


def process_single_job(request, configs, gpu_manager):
    """Process a single evaluation job and return the result."""
    work_args = deserialize_work_args(request['work_args'])
    kernel_src = request.get('kernel_src')
    kernel_name = request.get('kernel_name')
    run_name = request.get('run_name')
    run_dir = os.path.join(RUNS_DIR, run_name)
    configs.run_name = run_name
    os.makedirs(run_dir, exist_ok=True)

    logging.info(f"Processing evaluation for level {work_args.level}, problem {work_args.problem_id}, sample {work_args.sample_id}")

    # Acquire a free GPU
    gpu_id = gpu_manager.wait_for_gpu()
    work_args.device = torch.device(f"cuda:{gpu_id}")
    logging.info(f"Assigned GPU {gpu_id} to request")

    try:
        result = evaluate_single_sample_in_separate_process(
            work_args, configs, run_dir, kernel_src, kernel_name
        )
    finally:
        gpu_manager.release_gpu(gpu_id)
        logging.info(f"Released GPU {gpu_id}")
    return result


def handle_client(client_socket: socket.socket, configs, gpu_manager: 'GPUDeviceManager'):
    """Handle a single client connection, automatically detecting single or batch requests."""
    try:
        # Receive the request data
        data = b""
        while True:
            chunk = client_socket.recv(4096)
            if not chunk:
                break
            data += chunk
        if not data:
            return
        # Deserialize the request
        request = pickle.loads(data)

        # Auto-detect request type: if request has a 'batch' key, it's a batch request
        if 'batch' in request and isinstance(request['batch'], list):
            # Batch mode: request['batch'] is a list of job dicts
            start_time = time.time()
            job_list = request.get('batch', [])
            logging.info(f"Processing batch request with {len(job_list)} jobs")
            results = [None] * len(job_list)
            threads = []
            completed_count = 0
            progress_lock = threading.Lock()
            
            def run_job(idx, job):
                nonlocal completed_count
                try:
                    results[idx] = process_single_job(job, configs, gpu_manager)
                except Exception as e:
                    logging.info(f"Error in batch job: {e}")
                    results[idx] = KernelExecResult(
                        compiled=False, correctness=False,
                        metadata={"server_error": str(e)}
                    )
                finally:
                    with progress_lock:
                        completed_count += 1
                        pbar.update(1)
            
            with tqdm(total=len(job_list), desc="Processing batch jobs") as pbar:
                for idx, job in enumerate(job_list):
                    t = threading.Thread(target=run_job, args=(idx, job))
                    t.start()
                    threads.append(t)
                for t in threads:
                    t.join()
            end_time = time.time()
            logging.info(f"Batch request completed in {end_time - start_time:.2f} seconds")
            response_data = pickle.dumps(results)
            client_socket.sendall(response_data)
            client_socket.shutdown(socket.SHUT_WR)
        else:
            # Single job mode: request contains individual job data
            logging.info("Processing single job request")
            try:
                result = process_single_job(request, configs, gpu_manager)
            except Exception as e:
                logging.info(f"Error handling client: {e}")
                result = KernelExecResult(
                    compiled=False, correctness=False,
                    metadata={"server_error": str(e)}
                )
            response_data = pickle.dumps(result)
            client_socket.sendall(response_data)
            client_socket.shutdown(socket.SHUT_WR)
    finally:
        client_socket.close()


def main():
    config = parse_eval_server_args()

    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. Evaluation requires GPU.")

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    
    set_gpu_arch(config.gpu_arch)

     
    # Display GPU information
    logging.info("=" * 60)
    logging.info("GPU Information:")
    gpu_info = get_gpu_status_info()
    if "error" in gpu_info:
        logging.info(f"  {gpu_info['error']}")
    else:
        logging.info(f"  Total GPUs: {gpu_info['total_gpus']}")
        for gpu in gpu_info['gpu_details']:
            if 'error' in gpu:
                logging.info(f"  GPU {gpu['id']}: Error - {gpu['error']}")
            else:
                logging.info(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory_gb']:.1f} GB)")
    logging.info("=" * 60)
    
    logging.info(f"Starting evaluation server on port {config.port}")
    
    # Start the server
    start_evaluation_server(config.port, config)


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    main() 
