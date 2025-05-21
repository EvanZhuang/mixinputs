import logging
import os, types, importlib
from contextlib import ContextDecorator

# Set up logger
logger = logging.getLogger(__name__)

class NoCUDA(ContextDecorator):
    def __enter__(self):
        # # ---------- environment variable -------------------------------------
        self._old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""        # make CUDA runtime see 0 GPUs
        return self

    def __exit__(self, exc_type, exc, tb):
        # restore CUDA_VISIBLE_DEVICES
        if self._old_cuda_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._old_cuda_visible
        return False



def patch_vllm_llm():
    
    try:
        with NoCUDA():
            from .gpu_input_batch import InputBatch, CachedRequestState
            from .gpu_model_runner import GPUModelRunner

            gpu_runner = importlib.import_module("vllm.v1.worker.gpu_model_runner")
            input_batch = importlib.import_module("vllm.v1.worker.gpu_input_batch")
            
            if not hasattr(gpu_runner, 'original_GPUModelRunner'):
                # Store the original LLM init function
                original_gpu_model_runner = gpu_runner.GPUModelRunner
                original_gpu_input_batch = input_batch.InputBatch

                gpu_runner.original_GPUModelRunner = original_gpu_model_runner
                input_batch.original_InputBatch = original_gpu_input_batch

                # Patch it Up
                input_batch.InputBatch = InputBatch
                input_batch.CachedRequestState = CachedRequestState
                gpu_runner.GPUModelRunner = GPUModelRunner

                logger.debug("Successfully patched vllm")
            else:
                logger.debug("vllm LLM already patched")
            beta = os.environ.get("MIXINPUTS_BETA", None)
            if beta is not None:
                logger.info(f"MixInputs beta is set as {beta}")
            else:
                logger.info("MixInputs beta is using default value 1.0, set MIXINPUTS_BETA to change it.")
                os.environ["MIXINPUTS_BETA"] = "1.0"

        return True
        
    except Exception as e:
        logger.error(f"Error patching vllm LLM: {e}")
        # Attempt to restore original state if patching failed
        try:
            gpu_runner = importlib.import_module("vllm.v1.worker.gpu_model_runner")
            input_batch = importlib.import_module("vllm.v1.worker.gpu_input_batch")
            if hasattr(gpu_runner, 'original_GPUModelRunner'):
                gpu_runner.GPUModelRunner = gpu_runner.original_GPUModelRunner
                input_batch.InputBatch = input_batch.original_InputBatch
                logger.info("Restored original vllm classes after patching failure")
        except Exception as restore_error:
            logger.error(f"Error restoring original vllm classes: {restore_error}")
        return False