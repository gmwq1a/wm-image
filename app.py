# 万事屋图片去水印独立实例:IOPaint LaMa,512px 上限,每请求加载并释放
import time
from collections import defaultdict

import cv2
import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

ALLOWED_ORIGINS = [
    "https://wanshiwu.surge.sh",
    "https://geshi-gongfang.surge.sh",
    "https://wanshiwu-preview.onrender.com",
]
MAX_IMAGE_SIDE = 512
MAX_IMAGE_BYTES = 10 * 1024 * 1024
RATE_WINDOW = 60
RATE_IMAGE = 8

app = FastAPI(title="wm-image")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

_hits = defaultdict(list)


def rate_ok(ip, bucket, limit):
    now = time.time()
    key = f"{bucket}:{ip}"
    _hits[key] = [t for t in _hits[key] if now - t < RATE_WINDOW]
    if len(_hits[key]) >= limit:
        return False
    _hits[key].append(now)
    return True


def _load_model():
    import torch
    torch.set_num_threads(1)
    from iopaint.model_manager import ModelManager
    from iopaint.schema import Device
    return ModelManager(name="lama", device=Device.cpu)


def _unload_model(m):
    import gc
    del m
    gc.collect()
    try:
        from iopaint.helper import torch_gc
        torch_gc()
    except Exception:
        pass


@app.get("/")
@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}


@app.post("/api/image/dewatermark")
async def image_dewatermark(request: Request, image: UploadFile = File(...), mask: UploadFile = File(...)):
    ip = request.client.host if request.client else "?"
    if not rate_ok(ip, "img", RATE_IMAGE):
        return JSONResponse({"success": False, "error": "rate_limited"}, 429)
    img_bytes = await image.read()
    mask_bytes = await mask.read()
    if len(img_bytes) > MAX_IMAGE_BYTES:
        return JSONResponse({"success": False, "error": "image_too_large"}, 413)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    msk = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        return JSONResponse({"success": False, "error": "bad_image_or_mask"}, 400)
    h, w = img.shape[:2]
    scale = min(1.0, MAX_IMAGE_SIDE / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    _, msk = cv2.threshold(msk, 127, 255, cv2.THRESH_BINARY)
    msk = cv2.dilate(msk, np.ones((5, 5), np.uint8), iterations=1)
    from iopaint.schema import InpaintRequest
    model = _load_model()
    try:
        result = model(img, msk, InpaintRequest())  # output is BGR, encode directly
    finally:
        _unload_model(model)
    ok, buf = cv2.imencode(".png", result.astype(np.uint8))
    if not ok:
        return JSONResponse({"success": False, "error": "encode_failed"}, 500)
    return Response(content=buf.tobytes(), media_type="image/png")


