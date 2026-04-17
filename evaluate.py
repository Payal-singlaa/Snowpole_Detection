import torch
from torchvision.ops import box_iou

def evaluate(model, data_loader, device, iou_threshold=0.5, score_threshold=0.5):
    model.eval()

    TP = 0
    FP = 0
    FN = 0

    all_gt = []
    all_pred = []

    iou_list = []

    with torch.no_grad():
        for images, targets in data_loader:

            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(images)):
                pred_boxes = outputs[i]["boxes"].cpu()
                pred_scores = outputs[i]["scores"].cpu()
                pred_labels = outputs[i]["labels"].cpu()

                gt_boxes = targets[i]["boxes"]

                # Filter predictions
                keep = pred_scores > score_threshold
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]

                if len(pred_boxes) == 0:
                    FN += len(gt_boxes)
                    continue

                if len(gt_boxes) == 0:
                    FP += len(pred_boxes)
                    continue

                ious = box_iou(pred_boxes, gt_boxes)

                matched_gt = set()

                for p in range(len(pred_boxes)):
                    best_iou, gt_idx = torch.max(ious[p], dim=0)

                    if best_iou >= iou_threshold and gt_idx.item() not in matched_gt:
                        TP += 1
                        matched_gt.add(gt_idx.item())
                        iou_list.append(best_iou.item())
                    else:
                        FP += 1

                FN += len(gt_boxes) - len(matched_gt)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    return precision, recall, iou_list