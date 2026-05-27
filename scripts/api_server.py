import os
import sys
import argparse
import uvicorn
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field

from leakware.mlp import BinaryMlp


class DetectRequest(BaseModel):
    messages: List[dict] = Field(..., description="OpenAI format")
    prefill_type: str = Field("sys_prompt", description="prefill type: sys_prompt, rag_chunks, or universe")


class BatchDetectRequest(BaseModel):
    messages_list: List[List[dict]] = Field(..., description="batch of messages, each in OpenAI format")
    prefill_type: str = Field("sys_prompt", description="prefill type: sys_prompt, rag_chunks, or universe")


class DetectResponse(BaseModel):
    label: str = Field(..., description="detection result: attack 或 benign")
    probability: float = Field(..., description="attack probability (0~1)")
    threshold: float = Field(..., description="classification threshold")
    logprobs: List[float] = Field(..., description="logprobs of the prefill tokens")


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None



app = FastAPI(
    title="LeakAware",
    description="Eliciting LLM Leakage Awareness from Prefill Log Probabilities",
    version="0.0.1",
)

detector: BinaryMlp = None  # 全局检测器实例


@app.on_event("startup")
async def startup():
    global detector
    if detector is None:
        raise RuntimeError("detector is not initialized, please start the service with --model_path etc. arguments")


@app.get("/health", summary="health check")
async def health():
    return {"status": "ok"}


@app.get("/model/info", summary="model meta information")
async def model_info():
    return detector.get_model_info()


@app.post("/detect", response_model=DetectResponse, summary="single message leakage detection")
async def detect(req: DetectRequest):
    try:
        result = detector.detect(
            messages=req.messages,
            prefill_type=req.prefill_type,
        )
        return result
    except Exception as e:
        return {"error": str(e), "detail": "detection failed"}


@app.post("/detect/batch", summary="batch leakage detection")
async def detect_batch(req: BatchDetectRequest):
    try:
        results = detector.detect_batch(
            messages_list=req.messages_list,
            prefill_type=req.prefill_type,
        )
        return {"results": results, "total": len(results)}
    except Exception as e:
        return {"error": str(e), "detail": "batch detection failed"}


def main():
    global detector

    parser = argparse.ArgumentParser(description="leakSense API Server")
    parser.add_argument("--model_name", type=str, required=True, help="vLLM 模型名称")
    parser.add_argument("--base_url", type=str, required=True, help="vLLM 服务地址，如 http://127.0.0.1:22998/v1")
    parser.add_argument("--api_key", type=str, default="none", help="API Key")
    parser.add_argument("--reasoning_parser", type=str, default="deepseek_r1", choices=["deepseek_r1", "qwen3", "none"])
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务监听地址")
    parser.add_argument("--port", type=int, default=8900, help="服务端口")
    parser.add_argument("--device", type=str, default=None, help="推理设备 (cuda/cpu)，默认自动检测")
    args = parser.parse_args()

    detector = BinaryMlp(
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        reasoning_parser=args.reasoning_parser,
        device=args.device,
    )
    info = detector.get_model_info()
    print(f"模型加载完成: {info}")

    print(f"启动服务: http://{args.host}:{args.port}")
    print(f"Swagger 文档: http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
