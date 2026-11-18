
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image, UnidentifiedImageError

# Ultralytics replaces Image.open with a loader that tries to install an
# optional HEIF dependency for malformed files. Keep Pillow's original loader
# for request validation so a bad upload returns a normal client error.
PIL_IMAGE_OPEN = Image.open

from ultralytics import YOLO
from pathlib import Path
import shutil
from uuid import uuid4
from typing import Any

app = FastAPI(title="LUNA-SAFE AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best.pt"
UPLOAD_DIR = BASE_DIR / "backend" / "uploads"
FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL_PATH))


def saved_validation_metrics() -> dict[str, Any]:
    """Return validation metrics stored with the trained checkpoint.

    Object detection does not have one conventional accuracy value. mAP@50 is
    exposed as the headline detection score, alongside precision, recall, and
    the stricter mAP@50-95 score.
    """
    metrics = model.ckpt.get("train_metrics", {}) if isinstance(model.ckpt, dict) else {}

    def metric(key: str) -> float | None:
        value = metrics.get(key)
        return round(float(value), 4) if isinstance(value, (int, float)) else None

    return {
        "source": "Saved validation results from this model checkpoint",
        "headline_name": "mAP@50",
        "headline_score": metric("metrics/mAP50(B)"),
        "precision": metric("metrics/precision(B)"),
        "recall": metric("metrics/recall(B)"),
        "map50": metric("metrics/mAP50(B)"),
        "map50_95": metric("metrics/mAP50-95(B)"),
        "note": (
            "These are validation-set detection metrics recorded at training time, "
            "not a guarantee for an individual uploaded image."
        ),
    }


def build_ai_hazard_map(
    detections: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    size_threshold_px: float,
    density_threshold: int,
    grid_columns: int,
) -> dict[str, Any]:
    """Convert YOLO predictions into thresholded, image-relative hazard cells."""
    grid_rows = max(1, round(grid_columns * image_height / image_width))
    cell_width = image_width / grid_columns
    cell_height = image_height / grid_rows
    cells = [
        {
            "row": row,
            "column": column,
            "x": round(column * cell_width, 2),
            "y": round(row * cell_height, 2),
            "width": round(cell_width, 2),
            "height": round(cell_height, 2),
            "members": [],
        }
        for row in range(grid_rows)
        for column in range(grid_columns)
    ]

    for detection in detections:
        x, y = detection["center"]
        column = min(grid_columns - 1, max(0, int(x / cell_width)))
        row = min(grid_rows - 1, max(0, int(y / cell_height)))
        cells[row * grid_columns + column]["members"].append(detection)

    scored_cells = []
    for cell in cells:
        members = cell.pop("members")
        largest_diameter = max(
            (detection["size"]["diameter_px"] for detection in members),
            default=0.0,
        )
        size_score = min(60.0, largest_diameter / size_threshold_px * 60.0)
        density_score = min(40.0, len(members) / density_threshold * 40.0)
        hazard_score = round(size_score + density_score)
        # A terrain cell can only be a candidate-safe area when the AI found
        # no boulders in it. A small or isolated detected boulder is still a
        # caution area, even when it falls below the hazard-score thresholds.
        level = (
            "safe" if not members
            else "high" if hazard_score >= 70
            else "medium" if hazard_score >= 35
            else "caution"
        )
        scored_cells.append({
            **cell,
            "boulder_count": len(members),
            "largest_diameter_px": round(largest_diameter, 2),
            "hazard_score": hazard_score,
            "risk_level": level,
        })

    return {
        "source": "YOLO boulder predictions from this uploaded image",
        "method": "Size and local-density threshold scoring of AI detections",
        "thresholds": {
            "size_threshold_px": size_threshold_px,
            "density_threshold": density_threshold,
            "grid_columns": grid_columns,
            "grid_rows": grid_rows,
        },
        "cells": scored_cells,
    }


@app.get("/")
def home():
    return FileResponse(FRONTEND_FILE)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/model-report")
def model_report():
    """Expose the metrics embedded in best.pt without rerunning validation."""
    return saved_validation_metrics()


@app.post("/detect")
async def detect_boulders(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.25),
    size_threshold_px: float = Form(20),
    density_threshold: int = Form(2),
    grid_columns: int = Form(10),
):
    if not 0.01 <= confidence_threshold <= 0.99:
        raise HTTPException(
            status_code=422,
            detail="Confidence threshold must be between 0.01 and 0.99.",
        )
    if not 5 <= size_threshold_px <= 5000:
        raise HTTPException(
            status_code=422,
            detail="Boulder-size threshold must be between 5 and 5000 pixels.",
        )
    if not 1 <= density_threshold <= 100:
        raise HTTPException(
            status_code=422,
            detail="Density threshold must be between 1 and 100 boulders.",
        )
    if not 2 <= grid_columns <= 48:
        raise HTTPException(
            status_code=422,
            detail="Grid resolution must be between 2 and 48 columns.",
        )
    original_name = Path(file.filename or "").name
    suffix = Path(original_name).suffix.lower()

    if not original_name or suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail="Upload a PNG, JPEG, WebP, GIF, BMP, or TIFF image."
        )

    # A unique temporary name prevents two users uploading the same filename
    # from overwriting one another's image while inference is running.
    image_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"

    try:
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            with PIL_IMAGE_OPEN(image_path) as image:
                image_width, image_height = image.size
                image.verify()
        except Image.DecompressionBombError as error:
            raise HTTPException(
                status_code=422,
                detail=(
                    "This image is too large to analyze safely. "
                    "Upload a resized image or one of the tiles instead."
                )
            ) from error
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise HTTPException(
                status_code=422,
                detail="The uploaded file is not a readable image."
            ) from error

        results = model.predict(
            source=str(image_path),
            conf=confidence_threshold,
            imgsz=640
        )

        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)

                detections.append({
                    "class": "boulder",
                    "confidence": round(float(box.conf[0]), 3),
                    "box": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2)
                    ],
                    "center": [
                        round((x1 + x2) / 2, 2),
                        round((y1 + y2) / 2, 2),
                    ],
                    "size": {
                        "width_px": round(width, 2),
                        "height_px": round(height, 2),
                        "area_px2": round(width * height, 2),
                        "diameter_px": round((width * height) ** 0.5, 2),
                    },
                })

        return {
            "filename": original_name,
            "image_size": {"width_px": image_width, "height_px": image_height},
            "confidence_threshold": confidence_threshold,
            "total_boulders": len(detections),
            "detections": detections,
            "hazard_map": build_ai_hazard_map(
                detections,
                image_width,
                image_height,
                size_threshold_px,
                density_threshold,
                grid_columns,
            ),
            "model_report": saved_validation_metrics(),
        }
    finally:
        await file.close()
        image_path.unlink(missing_ok=True)
