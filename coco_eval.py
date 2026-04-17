import torch
import torchvision
from torchvision.ops import box_convert

def get_coco_evaluator(model, data_loader, device):
    model.eval()

    coco = torchvision.datasets.CocoDetection(
        img_folder=None,
        ann_file=None
    )

    coco_evaluator = torchvision.models.detection.coco_eval.CocoEvaluator(
        coco,
        iou_types=["bbox"]
    )

    with torch.no_grad():
        for images, targets in data_loader:

            images = [img.to(device) for img in images]

            outputs = model(images)

            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

            res = {}

            for i, output in enumerate(outputs):

                boxes = output["boxes"]
                scores = output["scores"]
                labels = output["labels"]

                res[i] = {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels
                }

            coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator