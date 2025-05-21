import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("MIXINPUTS_LOGGING_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def check_vllm_installed():
    """Check if vllm is installed"""
    try:
        import importlib.metadata
        importlib.metadata.version("vllm")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


# Check if patching is needed based on environment variables
if 'MIXINPUTS_BETA' in os.environ and check_vllm_installed():

    from .vllm_patch import patch_vllm_llm
    logger.info("Patching the installed vllm to enable Mixture of Inputs")
    # Patch the LLM class
    patch_status = patch_vllm_llm()
    logger.info(f"Patching vllm LLM... status: {patch_status}")
else:
    logger.debug("Skipping the patching of vllm, set MIXINPUTS_BETA to enable it.")