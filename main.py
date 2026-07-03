"""
Сервіс препроцесингу фото електролічильників для ТММ (двопрохідна схема).
Приймає фото + координати зони дисплея (від проходу 1, де Claude визначив, де дисплей).
Вирізає дисплей, збільшує, підвищує контраст, повертає покращений виріз (base64).

Координати очікуються в нормалізованій шкалі 0-1000 (як повертає Claude).
"""
import base64
import os

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="TMM Meter Preprocess", version="2.0")


class PreprocessRequest(BaseModel):
    image_base64: str          # вхідне фото у base64 (без префікса data:)
    # координати зони дисплея у шкалі 0-1000 (від проходу 1)
    x1: float
    y1: float
    x2: float
    y2: float
    padding: float = 0.03      # запас навколо зони (частка від розміру фото)
    scale: float = 4.0         # коефіцієнт збільшення вирізу
    clip_limit: float = 2.5    # сила CLAHE-контрасту
    sharpen: bool = True


class PreprocessResponse(BaseModel):
    image_base64: str
    width: int
    height: int


def crop_and_enhance(img, x1, y1, x2, y2, padding, scale, clip_limit, sharpen):
    h, w = img.shape[:2]
    # нормалізовані 0-1000 -> пікселі
    px1 = int(x1 / 1000.0 * w)
    py1 = int(y1 / 1000.0 * h)
    px2 = int(x2 / 1000.0 * w)
    py2 = int(y2 / 1000.0 * h)
    # запас
    pad_x = int(padding * w)
    pad_y = int(padding * h)
    px1 = max(0, px1 - pad_x)
    py1 = max(0, py1 - pad_y)
    px2 = min(w, px2 + pad_x)
    py2 = min(h, py2 + pad_y)
    # захист від виродженого прямокутника -> fallback на повне фото
    if px2 - px1 < 10 or py2 - py1 < 10:
        crop = img
    else:
        crop = img[py1:py2, px1:px2]

    # збільшення
    if scale and scale != 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # CLAHE-контраст через L-канал LAB
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l)
    crop = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # різкість
    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        crop = cv2.filter2D(crop, -1, kernel)

    return crop


@app.get("/")
def health():
    return {"status": "ok", "service": "meter-preprocess", "version": "2.0"}


@app.post("/preprocess", response_model=PreprocessResponse)
def preprocess(req: PreprocessRequest):
    try:
        raw = base64.b64decode(req.image_base64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("imdecode повернув None")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не вдалось декодувати: {e}")

    try:
        out = crop_and_enhance(img, req.x1, req.y1, req.x2, req.y2,
                               req.padding, req.scale, req.clip_limit, req.sharpen)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка обробки: {e}")

    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HTTPException(status_code=500, detail="Не вдалось закодувати результат")
    out_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    oh, ow = out.shape[:2]
    return PreprocessResponse(image_base64=out_b64, width=ow, height=oh)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
