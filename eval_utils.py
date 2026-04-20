import torch

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area


def evaluate_model(model, data_loader, device, iou_threshold=0.4, score_threshold=0.2):
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():
        for images, targets in data_loader:

            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(images)):
                preds = outputs[i]
                gt_boxes = targets[i]["boxes"].cpu().numpy()

                pred_boxes = preds["boxes"].cpu().numpy()
                scores = preds["scores"].cpu().numpy()

                matched = set()

                for pb, score in zip(pred_boxes, scores):

                    if score < score_threshold:
                        continue

                    found_match = False

                    for j, gt in enumerate(gt_boxes):
                        if j in matched:
                            continue

                        iou = compute_iou(pb, gt)

                        if iou >= iou_threshold:
                            total_tp += 1
                            matched.add(j)
                            found_match = True
                            break

                    if not found_match:
                        total_fp += 1

                total_fn += len(gt_boxes) - len(matched)

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)

    return precision, recall