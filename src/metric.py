
# Average accuracy across all pixels
def mean_pixel_accuracy(pred, target):
    correct = (pred == target).float()
    return correct.mean()  

# Intersection over Union (IoU) for a single class
def IoU(pred, target, num_classes):
    ious = []
    for i in range(num_classes):
        # Get predicted and target for the class i
        pred_class = (pred == i).float()
        target_class = (target == i).float()

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum() - intersection

        # Avoid division by zero
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)

    return ious  # List of IoU values for each class

# Dice coefficient for a single class
def dice_coeff(pred, target, num_classes):
    dice_scores = []
    for i in range(num_classes):
        # Get predicted and target for the class i
        pred_class = (pred == i).float()
        target_class = (target == i).float()

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()

        # Avoid division by zero
        if union == 0:
            dice_scores.append(float('nan'))
        else:
            dice_scores.append(2 * intersection / union)

    return dice_scores  # List of Dice scores for each class

# Pixel accuracy for each class
def per_class_accuracy(pred, target, num_classes):
    accs = []
    for i in range(num_classes):
        pred_class = (pred == i)
        target_class = (target == i)

        correct = (pred_class & target_class).sum().item()  # Count correctly predicted pixels
        total = target_class.sum().item()  # Count actual pixels of that class

        if total == 0:
            accs.append(float('nan'))  # Avoid division by zero
        else:
            accs.append(correct / total)  # Accuracy = correct / total pixels of that class

    return accs  # List of per-class accuracy values


