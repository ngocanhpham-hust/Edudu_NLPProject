from functools import lru_cache

from langchain_core.messages import HumanMessage

from src.config import settings


def _build_echo():
    class EchoLLM:
        def invoke(self, messages):
            prompt = messages[-1].content if messages else ""
            return type(
                "EchoResponse",
                (),
                {"content": "Echo backend đang bật. Hãy cấu hình LLM thật để sinh nội dung.\n\n" + prompt[:1200]},
            )()

    return EchoLLM()


def _build_hf_local():
    import torch
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(settings.hf_model)
    model = AutoModelForCausalLM.from_pretrained(settings.hf_model, torch_dtype=torch.bfloat16)
    text_gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=settings.hf_device,
        return_full_text=False,
        max_new_tokens=settings.hf_max_new_tokens,
        do_sample=settings.llm_temperature > 0,
        temperature=settings.llm_temperature,
    )
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=text_gen))


def _build_gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.google_api_key,
    )


def _build_vllm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.vllm_model,
        api_key=settings.vllm_api_key,
        base_url=settings.vllm_api_base,
        temperature=settings.llm_temperature,
    )


@lru_cache(maxsize=4)
def get_llm(provider: str | None = None):
    selected = provider or settings.llm_provider
    if selected == "echo":
        return _build_echo()
    if selected == "hf_local":
        return _build_hf_local()
    if selected == "gemini":
        return _build_gemini()
    if selected == "vllm":
        return _build_vllm()
    raise ValueError(f"Unknown llm_provider '{selected}'")


def invoke_llm(prompt: str, provider: str | None = None) -> str:
    response = get_llm(provider=provider).invoke([HumanMessage(content=prompt)])
    return response.content if isinstance(response.content, str) else str(response.content)
