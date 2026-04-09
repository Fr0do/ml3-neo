"""Starter for CV Detection Homework."""

import marimo
import os
import pickle
import torch
import torch.nn as nn
import torchvision.ops as ops

__generated_with = "0.10.0"
app = marimo.App()

@app.cell
def __():
    import marimo as mo
    return (mo,)

@app.cell
def __(nn):
    class YOLOHead(nn.Module):
        def __init__(self, in_channels: int, num_classes: int, num_anchors: int):
            super().__init__()
            # TODO: Реализовать regression head и classification head
            self.num_classes = num_classes
            self.num_anchors = num_anchors
            pass
            
        def forward(self, x):
            # TODO: Сделать forward pass
            pass
    return (YOLOHead,)

@app.cell
def __(torch, ops):
    def compute_iou(boxes1, boxes2):
        # TODO: Использовать torchvision.ops.box_iou
        return ops.box_iou(boxes1, boxes2)

    def apply_nms(boxes, scores, iou_threshold=0.5):
        # TODO: Использовать torchvision.ops.nms
        return ops.nms(boxes, scores, iou_threshold)
    return compute_iou, apply_nms

@app.cell
def __(os, pickle):
    # Anti-cheat block
    if os.environ.get("ML3_HIDDEN_TEST") == "1":
        print("Running hidden tests...")
        # TODO: load your model, run inference on test set
        predictions = [{"boxes": [], "scores": [], "labels": []}]
        
        os.makedirs("submission", exist_ok=True)
        with open("submission/predictions.pkl", "wb") as f:
            pickle.dump(predictions, f)
    return

if __name__ == "__main__":
    app.run()
