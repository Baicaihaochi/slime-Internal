from .math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy


def get_deepscaler_rule_based_reward(response, label, args=None, sample=None, evaluation=False):
    if "</think>" in response:
        model_solution = response.split("</think>")[-1]
    elif "###Response" in response:
        model_solution = response.split("###Response")[1]
    else:
        return 0

    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return 0
    if label == "":
        return 0

    # Convert single answer to list for uniform processing
    assert isinstance(label, (str, float, int))
    ground_truths = [label]

    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)

    if not processed_ground_truths:
        return 0

    # Check against all possible correct answers
    base_reward = 0
    for ground_truth in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
        if is_correct:
            base_reward = 1
            break
    # Apply response length penalty if enabled (but not during evaluation)
    final_reward = base_reward
    if (args and sample and hasattr(args, 'enable_length_penalty') and args.enable_length_penalty
        and not evaluation):  # Skip penalty during evaluation
        length_penalty = _compute_length_penalty_deepscaler(args, sample)
        final_reward = base_reward + length_penalty

        # Debug output when penalty is applied
        if length_penalty != 0:
            print(f"[DEEPSCALER REWARD DEBUG] Base: {base_reward}, Length penalty: {length_penalty:.3f}, Final: {final_reward:.3f}")
    elif evaluation and args and hasattr(args, 'enable_length_penalty') and args.enable_length_penalty:
        print(f"[DEEPSCALER EVAL] Length penalty disabled for evaluation - Base reward only: {base_reward}")

    return final_reward

def _compute_length_penalty_deepscaler(args, sample) -> float:
    """Compute response length penalty for deepscaler (same as DAPO).

    Args:
        args: Configuration arguments
        sample: Sample object containing response information

    Returns:
        Length penalty (non-positive value)
    """
    if not sample or not hasattr(sample, 'response_length'):
        return 0.0

    if not args.max_response_length:
        return 0.0

    response_length = sample.response_length
    max_length = args.max_response_length
    buffer_length = getattr(args, 'length_penalty_buffer', 1024)
    penalty_factor = getattr(args, 'length_penalty_factor', 1.0)

    # Calculate expected length (same logic as DAPO)
    expected_length = max_length - buffer_length

    if response_length <= expected_length:
        return 0.0  # No penalty for responses within expected length

    # Calculate penalty for responses exceeding expected length
    exceed_length = response_length - expected_length
    raw_penalty = -exceed_length / buffer_length * penalty_factor

    # Limit penalty to prevent extremely negative rewards
    # Max penalty should not exceed the base reward magnitude
    max_penalty = -1.0
    length_penalty = max(raw_penalty, max_penalty)

    return length_penalty