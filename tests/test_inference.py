from __future__ import annotations


def test_inference_module_imports():
    import inference
    assert hasattr(inference, "main")
