from __future__ import annotations

from collections.abc import Callable

from hallucination_eval.adapters.base import VLMAdapter

_REGISTRY: dict[str, Callable[[], VLMAdapter]] = {}


def register(name: str) -> Callable[[Callable[[], VLMAdapter]], Callable[[], VLMAdapter]]:
    def decorator(factory: Callable[[], VLMAdapter]) -> Callable[[], VLMAdapter]:
        _REGISTRY[name] = factory
        return factory

    return decorator


@register("llava-1.5-7b")
def _llava15() -> VLMAdapter:
    from hallucination_eval.adapters.llava_family import Llava15Adapter

    return Llava15Adapter()


@register("rs-llava")
def _rs_llava() -> VLMAdapter:
    from hallucination_eval.adapters.llava_family import RsLlavaAdapter

    return RsLlavaAdapter()


@register("geochat")
def _geochat() -> VLMAdapter:
    from hallucination_eval.adapters.geochat import GeoChatAdapter

    return GeoChatAdapter()


@register("qwen3-8b")
def _qwen3_vl() -> VLMAdapter:
    from hallucination_eval.adapters.qwen_family import Qwen3VLAdapter

    return Qwen3VLAdapter()


@register("qwen3.5-9b")
def _qwen35() -> VLMAdapter:
    from hallucination_eval.adapters.qwen_family import Qwen35Adapter

    return Qwen35Adapter()


def get_adapter(name: str) -> VLMAdapter:
    """Instantiate the adapter for a model key (e.g. llava-1.5-7b)."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def available() -> list[str]:
    return sorted(_REGISTRY)
