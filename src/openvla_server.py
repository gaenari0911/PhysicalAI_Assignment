import argparse
import base64
import io
import json
import os
import traceback
from pathlib import Path

# Reduce extra TensorFlow/backend noise.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor  # BitsAndBytesConfig 제거
import uvicorn

from typing import Optional


class PredictRequest(BaseModel):
    instruction: str
    image_b64: str
    unnorm_key: Optional[str] = None
    do_sample: bool = False


class OpenVLAServingModel:
    def __init__(self, model_path: str, device: str = "cuda", default_unnorm_key: str = "bridge_orig"):
        self.model_path = model_path
        self.device = device
        self.default_unnorm_key = default_unnorm_key

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # [수정] bitsandbytes 및 flash_attention_2 관련 설정을 모두 제거하고 순정 상태로 로드
        print("[INFO] Loading model in native bfloat16 with PyTorch SDPA...")
        self.vla = AutoModelForVision2Seq.from_pretrained(
            model_path,
            attn_implementation="sdpa",  # flash_attn 대신 파이토치 내장 고속 어텐션 사용
            torch_dtype=torch.bfloat16,   # 기본 정밀도 유지
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto"            # 메모리 상황에 맞게 자동 배치
        ) 

        stats_path = Path(model_path) / "dataset_statistics.json"
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                self.vla.norm_stats = json.load(f)
            print(f"[INFO] Loaded dataset statistics from: {stats_path}")
            print(f"[INFO] Available norm_stats keys: {list(self.vla.norm_stats.keys())}")
        else:
            print(f"[WARN] dataset_statistics.json not found at: {stats_path}")

    @torch.inference_mode()
    def predict(self, req: PredictRequest):
        image_bytes = base64.b64decode(req.image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        prompt = f"In: What action should the robot take to {req.instruction}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)

        unnorm_key = req.unnorm_key or self.default_unnorm_key

        action = self.vla.predict_action(
            **inputs,
            unnorm_key=unnorm_key,
            do_sample=req.do_sample,
        )

        if hasattr(action, "tolist"):
            action = action.tolist()

        action = [float(x) for x in action]
        if len(action) < 4:
            raise ValueError(f"Predicted action is too short: len={len(action)}, action={action}")

        print(f"[PREDICT] instruction={req.instruction}")
        print(f"[PREDICT] unnorm_key={unnorm_key}")
        print(f"[PREDICT] action={action}", flush=True)

        return {
            "action": action,
            "unnorm_key": unnorm_key,
            "prompt": prompt,
        }


def build_app(serving_model: OpenVLAServingModel):
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/predict")
    def predict(req: PredictRequest):
        try:
            return serving_model.predict(req)
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=str(exc))

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--default-unnorm-key", type=str, default="bridge_orig")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    serving_model = OpenVLAServingModel(
        model_path=args.model_path,
        device=args.device,
        default_unnorm_key=args.default_unnorm_key,
    )
    app = build_app(serving_model)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()