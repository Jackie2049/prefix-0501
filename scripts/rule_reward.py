"""Simple rule-based reward function for GRPO math training.

Checks if the response contains the correct numerical answer.
"""
import re

def compute_reward(data):
    """Compute reward based on whether the answer matches.

    Args:
        data: dict with 'response' (str) and 'extra_info' containing 'answer' (str)

    Returns:
        float: 1.0 if correct, 0.0 otherwise
    """
    response = data.get("response", "")
    answer = data.get("extra_info", {}).get("answer", "")

    try:
        # Extract the last number from the response
        numbers = re.findall(r'[\d.]+', response)
        if numbers:
            predicted = float(numbers[-1])
            target = float(answer)
            # Allow small tolerance
            return 1.0 if abs(predicted - target) < 0.5 else 0.0
    except (ValueError, TypeError):
        pass

    return 0.0