from pathlib import Path

import pytest

from chakra.src.converter.llm_converter import LLMConverter


_TRACE_HEADER = (
    "Layername comp_time input_loc input_size weight_loc weight_size "
    "output_loc output_size comm_type comm_size misc\n"
)
_LAYER = "layer_{index} 10 LOCAL 64 LOCAL 128 LOCAL 64 NONE 0 NONE\n"


def _trace(execution_type: str) -> str:
    if execution_type == "EVENT":
        prefix = "EVENT\n"
    else:
        prefix = f"{execution_type} model_parallel_NPU_group: 1\n"
    return prefix + "2\n" + _TRACE_HEADER + _LAYER.format(index=0) + _LAYER.format(index=1)


@pytest.mark.parametrize("execution_type", ["COLOCATED", "DECODE", "PREFILL", "EVENT"])
def test_convert_to_payloads_matches_legacy_files(tmp_path: Path, execution_type: str) -> None:
    trace_path = tmp_path / "trace.txt"
    trace_path.write_text(_trace(execution_type))

    legacy_prefix = tmp_path / "legacy"
    LLMConverter(str(trace_path), str(legacy_prefix), num_npus=2).convert()
    expected = {
        rank: (tmp_path / f"legacy.{rank}.et").read_bytes()
        for rank in range(4 if execution_type == "PREFILL" else 2)
    }

    memory_prefix = tmp_path / "memory"
    payloads = LLMConverter(
        str(trace_path), str(memory_prefix), num_npus=2
    ).convert_to_payloads()

    assert payloads == expected
    assert not list(tmp_path.glob("memory.*.et"))
