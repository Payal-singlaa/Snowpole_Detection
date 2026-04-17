def evaluate_map(model, data_loader, device):
    model.eval()

    scores = []
    labels = []

    with torch.no_grad():
        for images, targets in data_loader:

            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(images)):
                preds = outputs[i]

                scores.extend(preds["scores"].cpu().tolist())
                labels.extend([1] * len(preds["boxes"]))

    return scores, labels