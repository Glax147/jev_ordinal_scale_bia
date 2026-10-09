import pytest

from scripts.run_startlux_local import parse_answer, request_for


def test_startlux_question_omits_gold_and_preserves_choice_order():
    row = {
        "sample_id": "x", "context": "Example text", "question": "Choose one",
        "choices": ["low", "middle", "high"], "gold_position": 2,
    }
    state, questions = request_for(row)
    assert state == "Example text"
    assert questions == {"decision": {
        "type": "choice", "instructions": "Choose one",
        "criteria": {"a": "low", "b": "middle", "c": "high"},
    }}
    assert "gold_position" not in str(questions)


def test_startlux_answer_maps_key_to_position():
    row = {"sample_id": "x", "choices": ["low", "middle", "high"]}
    position, probabilities = parse_answer(row, {"decision": {
        "choice": "b", "probabilities": {"a": 0.1, "b": 0.7, "c": 0.2},
    }})
    assert position == 1
    assert probabilities == pytest.approx([0.1, 0.7, 0.2])
