from judge.format import _assignment_from_poly_epo_payload, build_judge_messages
from train.reward import compute_reward


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
