import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
import asyncio

class VllmGenerationLoop:
    def __init__(self, profile_config):
        # 初始化 vLLM 引擎
        engine_args = AsyncEngineArgs(
            model=profile_config["model_id"],
            quantization="fp8", # 或未來支援的 fp4
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=profile_config["hyperparameters"]["max_context_tokens"],
            enforce_eager=False, # 啟用 CUDA Graph 加速
            trust_remote_code=True
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # 讀取 Profile 中的動態停止詞 (防癲癇)
        self.stop_tokens = profile_config.get("stop_tokens", ["<|im_end|>"])
        
        self.sampling_params = SamplingParams(
            temperature=profile_config["hyperparameters"]["temperature"],
            top_p=profile_config["hyperparameters"]["top_p"],
            repetition_penalty=profile_config["hyperparameters"]["repetition_penalty"],
            stop=self.stop_tokens,
            max_tokens=profile_config["hyperparameters"]["max_generation_tokens"]
        )

    async def generate_stream(self, prompt: str, request_id: str):
        """非同步串流生成，支援中途斷路器攔截"""
        generator = self.engine.generate(prompt, self.sampling_params, request_id)
        full_text = ""
        
        async for request_output in generator:
            text = request_output.outputs[0].text
            delta = text[len(full_text):]
            full_text = text
            yield delta, full_text
            
            # 若 Tool Parser 發現符合工具調用語法，可透過 engine.abort(request_id) 提早中斷