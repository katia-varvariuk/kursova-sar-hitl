"""Демо-інтерфейс (Gradio): завантажити знімок і побачити рамки людей.

  python src/demo_app.py   ->  http://127.0.0.1:7860
"""
from __future__ import annotations

import gradio as gr
from PIL import ImageDraw

from predict import CONF_ACCEPT, predict

COLORS = {"auto_accept": (27, 175, 122), "human_check": (235, 104, 52)}


def run(image, use_sahi: bool, min_score: float):
    if image is None:
        return None, {"error": "завантажте зображення"}
    res = predict(image, sahi=use_sahi)
    res["boxes"] = [b for b in res["boxes"] if b["score"] >= min_score]
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for b in res["boxes"]:
        x, y, w, h = b["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=COLORS[b["route"]], width=3)
        draw.text((x + 2, max(y - 12, 0)), f"{b['score']:.2f}", fill=COLORS[b["route"]])
    return out, res


demo = gr.Interface(
    fn=run,
    inputs=[gr.Image(type="pil", label="Знімок з дрона"),
            gr.Checkbox(label="SAHI (нарізка 320, для дрібних людей)"),
            gr.Slider(0.05, 0.9, value=0.1, step=0.05, label="Показувати рамки з оцінкою від")],
    outputs=[gr.Image(type="pil", label=f"Зелені: auto_accept (≥ {CONF_ACCEPT}), помаранчеві: перевірити людині"),
             gr.JSON(label="Вихід predict()")],
    title="Пошук людей на знімках SAR (YOLO11n, після АМН на SARD)",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
