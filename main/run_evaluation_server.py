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
import torch
from configs import parse_eval_server_args, RUNS_DIR
from evaluation_utils import evaluate_single_sample_in_separate_process, KernelExecResult, deserialize_work_args


class GPUDeviceManager:
    """Thread-safe GPU device manager for the evaluation server"""
    
    def __init__(self, num_gpus: int):
        self.num_gpus = num_gpus
        self.available_gpus = set(range(num_gpus))
        self.lock = threading.Lock()
    
    def acquire_gpu(self) -> Optional[int]:
        """Acquire a free GPU device. Returns None if no GPU is available."""
        with self.lock:
            if self.available_gpus:
                gpu_id = self.available_gpus.pop()
                return gpu_id
            return None
    
    def release_gpu(self, gpu_id: int):
        """Release a GPU device back to the pool."""
        with self.lock:
            if 0 <= gpu_id < self.num_gpus:
                self.available_gpus.add(gpu_id)
    
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
        run_dir: Directory containing the run data
    
    The server expects requests with the following structure:
    {
        'work_args': EvaluationWorkArgs (serialized, device will be ignored),
        'kernel_src': str,
        'kernel_name': str
    }
    
    Returns evaluation results as KernelExecResult objects.
    """
    # Initialize GPU device manager
    num_gpus = torch.cuda.device_count()
    gpu_manager = GPUDeviceManager(num_gpus)
    print(f"Initialized GPU manager with {num_gpus} GPUs")
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind(('localhost', port))
        server_socket.listen(5)
        print(f"Evaluation server started on port {port}")
        
        while True:
            client_socket, address = server_socket.accept()
            print(f"Connection from {address}")
            
            # Handle each client in a separate thread
            client_thread = threading.Thread(
                target=handle_client,
                args=(client_socket, configs, gpu_manager)
            )
            client_thread.start()
            
    except KeyboardInterrupt:
        print("Server shutting down...")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server_socket.close()


def handle_client(client_socket: socket.socket, configs, gpu_manager: 'GPUDeviceManager'):
    """Handle a single client connection"""
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
        
        # Extract the components
        work_args = deserialize_work_args(request['work_args'])
        kernel_src = request.get('kernel_src')
        kernel_name = request.get('kernel_name')
        run_name = request.get('run_name')
        run_dir = os.path.join(RUNS_DIR, run_name)
        os.makedirs(run_dir, exist_ok=True) # make sure the run directory exists
        
        print(f"Processing evaluation for level {work_args.level}, problem {work_args.problem_id}, sample {work_args.sample_id}")
        
        # Acquire a free GPU
        gpu_id = gpu_manager.acquire_gpu()
        if gpu_id is None:
            print(f"No available GPU for request. Available: {gpu_manager.get_available_count()}, Used: {gpu_manager.get_used_count()}")
            error_result = KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={"server_error": "No available GPU devices"}
            )
            response_data = pickle.dumps(error_result)
            client_socket.sendall(response_data)
            return
        
        # Assign the acquired GPU to the work args
        work_args.device = torch.device(f"cuda:{gpu_id}")
        print(f"Assigned GPU {gpu_id} to request")
        
        try:
            # Run the evaluation
            result = evaluate_single_sample_in_separate_process(
                work_args, configs, run_dir, kernel_src, kernel_name
            )
        finally:
            # Always release the GPU, even if evaluation fails
            gpu_manager.release_gpu(gpu_id)
            print(f"Released GPU {gpu_id}")
        
        # Send the result back
        response_data = pickle.dumps(result)
        client_socket.sendall(response_data)
        
    except Exception as e:
        print(f"Error handling client: {e}")
        # Send error response
        error_result = KernelExecResult(
            compiled=False,
            correctness=False,
            metadata={"server_error": str(e)}
        )
        response_data = pickle.dumps(error_result)
        client_socket.sendall(response_data)
    finally:
        client_socket.close()


def main():
    config = parse_eval_server_args()
     
    # Display GPU information
    print("=" * 60)
    print("GPU Information:")
    gpu_info = get_gpu_status_info()
    if "error" in gpu_info:
        print(f"  {gpu_info['error']}")
    else:
        print(f"  Total GPUs: {gpu_info['total_gpus']}")
        for gpu in gpu_info['gpu_details']:
            if 'error' in gpu:
                print(f"  GPU {gpu['id']}: Error - {gpu['error']}")
            else:
                print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory_gb']:.1f} GB)")
    print("=" * 60)
    
    print(f"Starting evaluation server on port {config.port}")
    
    # Start the server
    start_evaluation_server(config.port, config)


if __name__ == "__main__":
    main() 