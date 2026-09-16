from typing import List, Tuple, Any
from pearl_agent.util import extract_code

def calculate_accuracy(predictions:list[str], problems: list[dict], verifier: Any) -> Tuple[float, List[str], List[int]]:
    """
    Calculates the zero-shot accuracy of code predictions.

    Args:
        predictions: A list of generated code strings, one for each problem.
        problems: The list of HumanEval problem dictionaries.
        verifier: The HumanEvalVerifier instance.

    Returns:
        A tuple containing:
        - accuracy (float): The fraction of correctly solved problems (i.e. passed all unit tests).
        - response (list[str]): A list of the verifier's `stdout` field, containing the results of each problem.
        - wrong (list[int]): A list of indices for the problems that failed.
    """
    num_correct = 0
    response = []
    incorrect_indices = []
    total = len(problems)
    for i, (pred, problem) in enumerate(zip(predictions, problems)):
        fn_name = problem["function_name"]
        test_suite = problem["test_suite"]
        res = verifier.verify(
            code=pred,
            function_name=fn_name,
            test_suite=test_suite,
        )
        response.append(res.get("stdout", ""))
        if res.get("passed_all", False):
            num_correct += 1
        else:
            incorrect_indices.append(i)
    accuracy = num_correct / total if total > 0 else 0.0
    return accuracy, response, incorrect_indices

def k_shot_acc(predictions: list[list[str]], problems: list[dict], verifier: Any) -> Tuple[float, List[str]]:
    """
    Calculates pass@k accuracy. A problem is solved if any of its k candidates pass.

    Args:
        predictions (list[list[str]]): A list where each item is another list of k code strings for a problem.
        problems (list[dict]): The list of HumanEval problem dictionaries.
        verifier (HumanEvalVerifier): The HumanEvalVerifier instance.

    Returns:
        A tuple containing:
        - accuracy (float): The pass@k accuracy.
        - solved_fns (list[str]): A list of function names for problems that were solved.
    """
    num_probs = len(predictions)
    num_corr = 0
    solved_fns = []

    for candidates, problem in zip(predictions, problems):
        fn_name = problem["function_name"]
        test_suite = problem["test_suite"]
        for candidate in candidates:
            res = verifier.verify(
                code=candidate,
                function_name=fn_name,
                test_suite=test_suite,
            )
            if res.get("passed_all", False):
                solved_fns.append(fn_name)
                num_corr += 1
                break
    accuracy = num_corr / num_probs if num_probs > 0 else 0.0
    return accuracy, solved_fns