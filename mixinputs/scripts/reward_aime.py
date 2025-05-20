import re
import random
import ast
import operator


def extract_solution(solution_str):
    """Extract the integer from the solution string."""
    #solution_str = solution_str.split('\n\n')[-1]
    solution_str = solution_str.replace('\n', ' ')
    
    # Define patterns for both <answer>...</answer> and \boxed{...}
    answer_patterns = [
        r'\\boxed{(.*?)}'         # Matches \boxed{...} (escaped for regex)
    ]
    
    final_answer = None
    for pattern in answer_patterns:
        matches = list(re.finditer(pattern, solution_str))
        if matches:
            final_answer = matches[-1].group(1).strip()  # Extract the last match
            # check if the final answer is a valid expression
            try:
                # Use ast.literal_eval to safely evaluate the expression
                final_answer = ast.literal_eval(final_answer)
                if isinstance(final_answer, (int, float)):
                    return final_answer
            except (ValueError, SyntaxError):
                pass
    # If no matches found, return None
    return None