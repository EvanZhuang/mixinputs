import argparse
import re, json
from datasets import load_dataset
from vllm import LLM, SamplingParams
import transformers
import os
import torch

from reward import compute_score, extract_solution, validate_equation

def parse_arguments():
    """
    Parse command-line arguments for the AIME evaluation script.
    """
    parser = argparse.ArgumentParser(description="Evaluate a (cutting-edge) LLM on the AIME dataset using vLLM.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Hugging Face model name or path (e.g. 'gpt2', 'EleutherAI/gpt-neo-1.3B', etc.)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="yzhuang/tinyzero-Countdown-Tasks-4",
        help="Hugging Face dataset name or path (e.g. 'yzhuang/tinyzero-Countdown-Tasks-4')"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "validation"],
        help="Which dataset split to evaluate on"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate for each response"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (0.0 = greedy decoding)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--eager",
        type=str,
        default="False",
        help="Whether to use eager execution (default: False)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (default: False)"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (default: False)"
    )
    args = parser.parse_args()
    return args

def extract_answer_integer(generated_text: str):
    """
    Attempt to extract the integer answer from the LLM's output using multiple heuristics:
      1. Look for a pattern like 'final answer is' or 'the answer is'.
      2. Otherwise, try looking in the last line for an integer.
      3. Finally, default to searching for the first integer in the entire text.

    Returns an integer if found, else None.
    """

    # 1. Look for a phrase like "final answer is X" or "the answer is X", or \\boxed{X}.
    #    You can add more patterns if needed.
    patterns = [
        # r"(?:final answer is|the final answer is|the answer is)\s+(-?\d+)",
        # r"(?:answer:\s*)(-?\d+)",
        # r"\\boxed{(-?\d+)}",
        r"\\boxed{(-?\d+)}",
    ]
    for pattern in patterns:
        match = re.search(pattern, generated_text, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass  # fallback to other methods if this fails

    # 2. Try to parse an integer from the last line of text
    last_line = generated_text.strip().split('\n')[-1]
    match = re.search(r"-?\d+", last_line)
    if match:
        try:
            return int(match.group())
        except ValueError:
            pass

    # 3. Fallback: parse the first integer found anywhere in the text
    match = re.search(r"-?\d+", generated_text)
    if match:
        try:
            return int(match.group())
        except ValueError:
            pass

    # If nothing found, return None
    return None

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # 1. Load the dataset
    dataset = load_dataset(args.dataset_name)['train']
    dataset = dataset.select(range(128))
    print("Loaded {} examples from dataset {}".format(len(dataset), args.dataset_name))

    model_base_name = args.model_name.split("/")[-1]
    dataset_base_name = args.dataset_name.split("/")[-1]
    # create folder if not exists

    os.makedirs(args.output_dir, exist_ok=True)
    file_name = f"{model_base_name}={dataset_base_name}={args.top_p}={args.temperature}.json"
    # Check if file already exists
    if os.path.exists(f"{args.output_dir}/{file_name}") and not args.debug:
        print(f"File {args.output_dir}/{file_name} already exists. Skipping evaluation.")
        return

    # 2. Initialize the vLLM LLM
    MAX_MODEL_LEN = args.max_new_tokens + 1024
    llm = LLM(args.model_name, dtype="bfloat16", tensor_parallel_size=torch.cuda.device_count(), max_model_len=MAX_MODEL_LEN, max_num_seqs=128, enable_chunked_prefill=False, enforce_eager=args.eager.lower() == "true", gpu_memory_utilization=0.95, trust_remote_code=True) 
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # 3. Prepare generation parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
    )

    correct = 0
    total = 0

    # OPTIONAL: If you want to speed things up with batching, you can collect 
    # the dataset questions in batches of size args.batch_size.
    # For simplicity here, we'll show the standard per-example loop.

    # 4. Inference loop
    all_prompts = []
    gold_answers = []
    predicted_solutions = []
    predictions = []

    for example in dataset:
        prompt = example["prompt"][0]
        # Speicific for these tiny zero datasets
        generation_prompt = prompt['content'].split("Assistant:")[1].strip()
        prompt['content'] = prompt['content'].split("Assistant:")[0]

        gold_answer = example["reward_model"]['ground_truth']  # integer
        gold_answers.append(gold_answer)

        if "nemotron" in args.model_name.lower():
            system_prompt = {"role": "system", "content": "detailed thinking on"}
            prompt['content'] = prompt['content'] + "No latex symbols in the final answer! Double check the final answer."
            prompt = tokenizer.apply_chat_template([system_prompt, prompt], tokenize=False, add_generation_prompt=True)
        else:
            prompt = tokenizer.apply_chat_template([prompt], tokenize=False, add_generation_prompt=True) + generation_prompt
        all_prompts.append(prompt)
    
    # We can process in batches if desired (to speed up generation).
    # Here’s a simple example of chunking the prompts into smaller batches:
    outputs = llm.generate(all_prompts, sampling_params)

    # Each element in 'outputs' corresponds to one input prompt
    for out, gold in zip(outputs, gold_answers):
        # out is a RequestOutput containing .outputs (list of model generations)
        generation = out.outputs[0].text  # we only requested n=1
        predicted_solutions.append(generation)
        # remove the "," in between numbers 
        generation = re.sub(r'(\d+),(\d+)', r'\1\2', generation)
        if type(gold) == dict:
            score = compute_score(generation, gold)
            if score > 0.1: correct += 1 #More than format
            predictions.append(extract_solution(generation))
        else:
            # Extract integer from the generated text
            pred_answer = extract_answer_integer(generation)
            predictions.append(pred_answer)
            if pred_answer is not None and int(pred_answer) == int(gold):
                correct += 1
        total += 1

    # 5. Print results
    accuracy = correct / total if total > 0 else 0
    print(f"Evaluation on split='{args.split}' with model='{args.model_name}':")
    print(f"  Total questions: {total}")
    print(f"  Correct answers: {correct}")
    print(f"  Accuracy: {accuracy:.2%}")

    # Save the predictions to a file
    out_json = [{"accuracy": accuracy, "total": total, "correct": correct}]
    out_json = out_json + [{"question": q,"gold": g, "predicted": p, "pred_solution": s} for q, g, p, s in zip(all_prompts, gold_answers, predictions, predicted_solutions)]

    with open(f"{args.output_dir}/{file_name}", "w") as f:
        json.dump(out_json, f, indent=2)

    print(f"Results saved to {args.output_dir}/{file_name}")

if __name__ == "__main__":
    main()