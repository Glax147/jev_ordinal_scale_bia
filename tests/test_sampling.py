from jev_bias.sampling import hash_stratified_sample


def test_hash_sampling_is_input_order_independent():
    rows = [
        {"sample_id": f"d::{i}", "source": "d", "sampling_stratum": str(i % 2),
         "question": "q", "context": f"context {i}"}
        for i in range(20)
    ]
    a = [row["sample_id"] for row in hash_stratified_sample(rows, 8, 42)]
    b = [row["sample_id"] for row in hash_stratified_sample(reversed(rows), 8, 42)]
    assert a == b
    assert len(a) == 8


def test_content_dedup_is_input_order_independent():
    rows = [
        {"sample_id": "b", "source": "d", "sampling_stratum": "0", "question": "same", "context": "text"},
        {"sample_id": "a", "source": "d", "sampling_stratum": "0", "question": "same", "context": "text"},
        {"sample_id": "c", "source": "d", "sampling_stratum": "1", "question": "other", "context": "text"},
    ]
    a = [row["sample_id"] for row in hash_stratified_sample(rows, 2, 42)]
    b = [row["sample_id"] for row in hash_stratified_sample(reversed(rows), 2, 42)]
    assert a == b


def test_duplicate_sample_ids_are_rejected():
    rows = [
        {"sample_id": "a", "source": "d", "sampling_stratum": "0", "question": "q1", "context": "c1"},
        {"sample_id": "a", "source": "d", "sampling_stratum": "1", "question": "q2", "context": "c2"},
    ]
    import pytest

    with pytest.raises(ValueError, match="Duplicate sample_id"):
        hash_stratified_sample(rows, 1, 42)


def test_candidates_are_part_of_content_signature():
    rows = [
        {"sample_id": "a", "source": "d", "sampling_stratum": "0", "question": "q", "context": "c",
         "choices": ["x", "y"], "gold_position": 0},
        {"sample_id": "b", "source": "d", "sampling_stratum": "0", "question": "q", "context": "c",
         "choices": ["x", "z"], "gold_position": 0},
    ]
    assert len(hash_stratified_sample(rows, 2, 42)) == 2


def test_conflicting_duplicate_labels_are_rejected():
    import pytest

    rows = [
        {"sample_id": "a", "source": "d", "sampling_stratum": "0", "question": "q", "context": "c",
         "choices": ["x", "y"], "gold_position": 0},
        {"sample_id": "b", "source": "d", "sampling_stratum": "0", "question": "q", "context": "c",
         "choices": ["x", "y"], "gold_position": 1},
    ]
    with pytest.raises(ValueError, match="Conflicting labels"):
        hash_stratified_sample(rows, 1, 42)


def test_sources_with_same_stratum_are_allocated_separately():
    rows = [
        {"sample_id": f"{source}-{i}", "source": source, "sampling_stratum": "shared",
         "question": f"q{i}", "context": f"c{i}", "gold_position": 0}
        for source in ("a", "b") for i in range(3)
    ]
    selected = hash_stratified_sample(rows, 2, 42)
    assert {row["source"] for row in selected} == {"a", "b"}


def test_null_gold_text_falls_back_to_position_for_conflict_detection():
    import pytest

    rows = [
        {"sample_id": "a", "source": "d", "sampling_stratum": "0", "question": "q", "context": "c",
         "choices": ["x", "y"], "gold_label_text": None, "gold_position": 0},
        {"sample_id": "b", "source": "d", "sampling_stratum": "0", "question": "q", "context": "c",
         "choices": ["x", "y"], "gold_label_text": None, "gold_position": 1},
    ]
    with pytest.raises(ValueError, match="Conflicting labels"):
        hash_stratified_sample(rows, 1, 42)


def test_source_allocation_precedes_within_source_strata():
    rows = []
    for i in range(30):
        rows.append({"sample_id": f"a-{i}", "source": "a", "sampling_stratum": "only",
                     "question": f"qa{i}", "context": f"ca{i}", "gold_position": 0})
        rows.append({"sample_id": f"b-{i}", "source": "b", "sampling_stratum": str(i % 5),
                     "question": f"qb{i}", "context": f"cb{i}", "gold_position": 0})
    selected = hash_stratified_sample(rows, 20, 42)
    from collections import Counter

    assert Counter(row["source"] for row in selected) == {"a": 10, "b": 10}
