DAPO_PROMPT_TEMPLATE = """Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:"."""

VERL_MATH_BOXED_TEMPLATE = (
    "{problem} Let's think step by step and output the final answer within \\boxed{{}}."
)

VERL_POLARIS_MAXRL_TEMPLATE = (
    "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
)

HYBRID_ANSWER_BOXED_TEMPLATE = """Solve the following math problem step by step. End your response with the final answer on its own line, formatted exactly as: Answer: \\boxed{{$Answer}}

{problem}

Remember: the last line must be "Answer: \\boxed{{...}}" with your final answer inside the box."""

PROMPT_VARIANTS = {
    "dapo_answer_v1": DAPO_PROMPT_TEMPLATE,
    "verl_math_boxed": VERL_MATH_BOXED_TEMPLATE,
    "verl_polaris_maxrl": VERL_POLARIS_MAXRL_TEMPLATE,
    "hybrid_answer_boxed": HYBRID_ANSWER_BOXED_TEMPLATE,
}


def format_problem(problem: str, variant: str = "dapo_answer_v1") -> str:
    return PROMPT_VARIANTS[variant].format(problem=problem)
