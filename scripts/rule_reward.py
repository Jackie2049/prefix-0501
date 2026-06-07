"""Rule-based reward function for GRPO math training.

Matches verl's compute_score interface:
  compute_score(data_source, solution_str, ground_truth, extra_info)

Checks if the response contains the correct numerical answer.
"""
import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Compute reward based on whether the answer matches.

    Args:
        data_source: str identifying the data source (e.g. "math")
        solution_str: decoded response text from the model
        ground_truth: expected answer string
        extra_info: dict with optional metadata

    Returns:
        float: 1.0 if correct, 0.0 otherwise
    """
    try:
        # Extract the last number from the response
        numbers = re.findall(r'[\d.]+', solution_str)
        if numbers:
            predicted = float(numbers[-1])
            target = float(ground_truth)
            # Allow small tolerance
            return 1.0 if abs(predicted - target) < 0.5 else 0.0
    except (ValueError, TypeError):
        pass

    return 0.0