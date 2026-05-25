from judge.format import _assignment_from_poly_epo_payload, build_judge_messages
from train.prompts import PROMPT_VARIANTS, format_problem
from train.reward import compute_reward, extract_rank2


def test_compute_reward_correct_answer():
    result = compute_reward("step by step\nAnswer: 42", "42")
    assert result["reward"] == 1
    assert result["parse_ok"] is True


def test_compute_reward_wrong_answer():
    result = compute_reward("step by step\nAnswer: 41", "42")
    assert result["reward"] == 0
    assert result["parse_ok"] is True


def test_compute_reward_boxed_only():
    result = compute_reward("step by step\n\\boxed{42}", "42")
    assert result["reward"] == 0
    assert result["parse_ok"] is False


def test_compute_reward_no_answer_marker():
    result = compute_reward("step by step with no final line", "42")
    assert result["reward"] == 0
    assert result["parse_ok"] is False


def test_format_problem_variants():
    problem = "What is 2+2?"
    assert "Answer:" in format_problem(problem, variant="dapo_answer_v1")
    assert "\\boxed{}" in format_problem(problem, variant="verl_math_boxed")
    assert "Answer: \\boxed{" in format_problem(problem, variant="hybrid_answer_boxed")
    assert set(PROMPT_VARIANTS) == {
        "dapo_answer_v1",
        "verl_math_boxed",
        "hybrid_answer_boxed",
    }


def test_extract_rank2_minerva_path():
    result = extract_rank2("work\nAnswer: 42", "42")
    assert result["parse_ok_minerva"] is True
    assert result["parsed_answer_minerva"] == "42"
    assert result["extract_path"] == "answer_line"
    assert result["parse_ok_rank2"] is True
    assert result["reward"] == 1


def test_extract_rank2_boxed_path():
    result = extract_rank2("work\n\\boxed{42}", "42", prompt_variant="verl_math_boxed")
    assert result["parse_ok_boxed"] is True
    assert result["parsed_answer_boxed"] == "42"
    assert result["extract_path"] == "boxed"
    assert result["parse_ok_rank2"] is True
    assert result["reward"] == 1


def test_extract_rank2_hybrid_path():
    completion = "work\nAnswer: \\boxed{42}"
    result = extract_rank2(
        completion, "42", prompt_variant="hybrid_answer_boxed"
    )
    assert result["extract_path"] == "hybrid"
    assert result["parse_ok_rank2"] is True
    assert result["parsed_answer"] == "42"
    assert result["reward"] == 1


def test_extract_rank2_order_hybrid_before_boxed():
    completion = "Answer: \\boxed{7}\nalso \\boxed{99}"
    result = extract_rank2(
        completion, "7", prompt_variant="hybrid_answer_boxed"
    )
    assert result["extract_path"] == "hybrid"
    assert result["parsed_answer"] == "7"
    assert result["reward"] == 1


def test_extract_rank2_order_boxed_before_minerva():
    completion = "Answer: 99\n\\boxed{42}"
    result = extract_rank2(completion, "42", prompt_variant="verl_math_boxed")
    assert result["extract_path"] == "boxed"
    assert result["parsed_answer"] == "42"
    assert result["reward"] == 1


def test_build_judge_messages_eight_rollouts():
    rollouts = [{"completion": f"resp{i}"} for i in range(8)]
    system, user = build_judge_messages("What is 2+2?", rollouts)
    assert "8" in system
    lines = user.splitlines()
    for n in range(1, 9):
        assert any(line.startswith(f"{n}.") for line in lines)


def test_build_judge_messages_two_rollouts_smoke():
    rollouts = [{"completion": "first"}, {"completion": "second"}]
    system, user = build_judge_messages("What is 2+2?", rollouts)
    assert "2" in system
    lines = user.splitlines()
    assert any(line.startswith("1.") for line in lines)
    assert any(line.startswith("2.") for line in lines)
    assert not any(line.startswith("3.") for line in lines)


def test_assignment_cluster_id_100_degenerate():
    payload = {
        str(i): {
            "chain_of_thought": "Macro: x. Micro: y.",
            "cluster_id": 100 if i == 1 else 0,
        }
        for i in range(1, 3)
    }
    assignment, _clusters = _assignment_from_poly_epo_payload(payload, n_responses=2)
    assert assignment[0] == -1
    assert assignment[1] == 0
