import copy

import pytest

from uni_api.upstream.responses_normalization import (
    ResponsesCustomToolCallIdCollisionError,
    ResponsesCustomToolCallIdNormalizer,
    responses_custom_tool_call_id_normalization_enabled,
)


def test_normalizes_only_top_level_custom_tool_call_input_items():
    payload = {
        "input": [
            {
                "type": "reasoning",
                "id": "item_reasoning123",
                "content": [
                    {
                        "type": "custom_tool_call",
                        "id": "item_nested123",
                    }
                ],
            },
            {
                "type": "custom_tool_call",
                "id": "ctc_alreadycanonical123",
                "call_id": "call_existing",
            },
            {
                "type": "custom_tool_call",
                "id": "item_7eb1bd749e0a9e692c69ed40",
                "call_id": "call_foCUR1DBzdZeYyOccLpOmwUF",
            },
            {
                "type": "custom_tool_call_output",
                "id": "ctco_output123",
                "call_id": "call_foCUR1DBzdZeYyOccLpOmwUF",
            },
        ]
    }

    normalizer = ResponsesCustomToolCallIdNormalizer()
    result = normalizer.normalize(payload)

    assert payload["input"][0]["id"] == "item_reasoning123"
    assert payload["input"][0]["content"][0]["id"] == "item_nested123"
    assert payload["input"][1]["id"] == "ctc_alreadycanonical123"
    assert payload["input"][2] == {
        "type": "custom_tool_call",
        "id": "ctc_7eb1bd749e0a9e692c69ed40",
        "call_id": "call_foCUR1DBzdZeYyOccLpOmwUF",
    }
    assert payload["input"][3]["call_id"] == "call_foCUR1DBzdZeYyOccLpOmwUF"
    assert result.normalized_ids == 1
    assert result.rewritten_references == 0
    assert result.paths == ("input[2].id",)

    second_result = normalizer.normalize(payload)
    assert not second_result.changed


def test_non_alphanumeric_item_suffix_is_not_normalized():
    payload = {
        "input": [
            {
                "type": "custom_tool_call",
                "id": "item_not-canonical",
                "call_id": "call_1",
            }
        ]
    }

    result = ResponsesCustomToolCallIdNormalizer().normalize(payload)

    assert not result.changed
    assert payload["input"][0]["id"] == "item_not-canonical"


def test_collision_is_rejected_before_payload_mutation():
    payload = {
        "input": [
            {
                "type": "custom_tool_call",
                "id": "item_duplicate123",
                "call_id": "call_1",
            },
            {
                "type": "custom_tool_call",
                "id": "ctc_duplicate123",
                "call_id": "call_2",
            },
        ]
    }
    original = copy.deepcopy(payload)

    with pytest.raises(
        ResponsesCustomToolCallIdCollisionError,
        match="would collide with an existing item ID",
    ):
        ResponsesCustomToolCallIdNormalizer().normalize(payload)

    assert payload == original


def test_collision_with_item_from_previous_stream_event_is_rejected():
    normalizer = ResponsesCustomToolCallIdNormalizer()
    normalizer.normalize(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "custom_tool_call",
                "id": "ctc_duplicate123",
                "call_id": "call_existing",
            },
        }
    )
    payload = {
        "type": "response.output_item.added",
        "output_index": 1,
        "item": {
            "type": "custom_tool_call",
            "id": "item_duplicate123",
            "call_id": "call_new",
        },
    }
    original = copy.deepcopy(payload)

    with pytest.raises(
        ResponsesCustomToolCallIdCollisionError,
        match="would collide with an existing item ID",
    ):
        normalizer.normalize(payload)

    assert payload == original


def test_stream_event_sequence_uses_one_consistent_id_mapping():
    normalizer = ResponsesCustomToolCallIdNormalizer()
    item = {
        "type": "custom_tool_call",
        "id": "item_stream123",
        "call_id": "call_stream123",
        "name": "exec",
        "input": "{}",
    }
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": copy.deepcopy(item),
        },
        {
            "type": "response.custom_tool_call_input.delta",
            "output_index": 0,
            "item_id": "item_stream123",
            "delta": "{}",
        },
        {
            "type": "response.custom_tool_call_input.done",
            "output_index": 0,
            "item_id": "item_stream123",
            "input": "{}",
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": copy.deepcopy(item),
        },
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [copy.deepcopy(item)],
            },
        },
    ]

    results = [normalizer.normalize(event) for event in events]

    assert events[0]["item"]["id"] == "ctc_stream123"
    assert events[1]["item_id"] == "ctc_stream123"
    assert events[2]["item_id"] == "ctc_stream123"
    assert events[3]["item"]["id"] == "ctc_stream123"
    assert events[4]["response"]["output"][0]["id"] == "ctc_stream123"
    assert [result.normalized_ids for result in results] == [1, 0, 0, 1, 1]
    assert [result.rewritten_references for result in results] == [0, 1, 1, 0, 0]


@pytest.mark.parametrize(
    ("configured", "models", "expected"),
    [
        (True, ("gpt-5.6-sol",), True),
        (False, ("gpt-5.6-sol",), False),
        (["gpt-5.6-sol"], ("gpt-5.6-sol",), True),
        (["gpt-5.6-sol"], ("gpt-5.6-terra",), False),
        (["*"], ("gpt-5.6-terra",), True),
        ("gpt-5.6-sol", ("gpt-5.6-sol",), False),
    ],
)
def test_provider_model_feature_flag(configured, models, expected):
    provider = {
        "preferences": {
            "normalize_responses_custom_tool_call_ids": configured,
        }
    }

    assert (
        responses_custom_tool_call_id_normalization_enabled(provider, models)
        is expected
    )


@pytest.mark.parametrize(
    ("provider_name", "model", "expected"),
    [
        ("fugue-codex", "gpt-5.6-sol", True),
        ("937auth", "gpt-5.6-sol", True),
        ("937auth01", "gpt-5.6-sol", True),
        ("fugue-codex", "gpt-5.6-terra", False),
        ("unrelated", "gpt-5.6-sol", False),
    ],
)
def test_default_provider_model_compatibility_matrix(provider_name, model, expected):
    provider = {"provider": provider_name, "preferences": {}}

    assert (
        responses_custom_tool_call_id_normalization_enabled(provider, (model,))
        is expected
    )


def test_provider_setting_can_disable_default_compatibility_matrix():
    provider = {
        "provider": "fugue-codex",
        "preferences": {
            "normalize_responses_custom_tool_call_ids": False,
        },
    }

    assert not responses_custom_tool_call_id_normalization_enabled(
        provider,
        ("gpt-5.6-sol",),
    )
