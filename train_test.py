import os
import torch
import numpy as np
import copy
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import sys
sys.path.append('GroundingDINO')
from groundingdino.util.base_api import load_model, threshold
import os
import numpy as np
from datetime import datetime

from utils.processor import DataProcessor
from utils.criterion import SetCriterion
from utils.image_loader import get_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
TEXT_TRESHOLD = 0.25

""" data """
processor = DataProcessor()
annotations = processor.annotations

BATCH_SIZE = 1
train_loader = get_loader(processor, 'train', BATCH_SIZE)
val_loader = get_loader(processor, 'val', BATCH_SIZE)
test_loader = get_loader(processor, 'test', BATCH_SIZE)

loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
print("Data loaded!")
print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")


""" model"""
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
model = model.to(device)

# freeze encoders
for param in model.backbone.parameters():
    param.requires_grad = False
for param in model.bert.parameters():
    param.requires_grad = False


""" criterion """
criterion = SetCriterion()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0001)



def train(epoch):
    print(f"Training on train set data")
    model.train()
    loader = loaders['train']

    train_mae = 0
    train_rmse = 0
    
    train_tp = 0
    train_fp = 0
    train_fn = 0
    
    counter = 0
    counter_for_image = 0
    train_size = len(loader.dataset) 


    for images, captions, shapes, img_caps in loader: # tensor, list of list [caption] for each image in the batch, list, list of list [(img, cap)] for each img in the batch
        # images: [b1_img, b2_img,...] captions: [ [b1_cap1, b1_cap2], [b2_cap1, b2_cap2], ...]

        mask_bi = [i for i, img_cap_list in enumerate(img_caps) for _ in img_cap_list] # index for each img,cap pair in the batch
        anno_b = [annotations[img_cap] for img_cap_list in img_caps for img_cap in img_cap_list] 
        img_caps = [img_cap for img_cap_list in img_caps for img_cap in img_cap_list]
        shapes = [shapes[i] for i, caption_list in enumerate(captions) for _ in caption_list]
        
        optimizer.zero_grad()

        
        # duplicate each image number of times that is equal to the number of captions for that image
        images = torch.stack([images[i] for i, caption_list in enumerate(captions) for _ in caption_list], dim=0)
        captions = [caption for caption_list in captions for caption in caption_list] # flatten list of list
        images = images.to(device)
        outputs = model(images, captions=captions)

        
        outputs["pred_points"] = outputs["pred_boxes"][:, :, :2] 
        
        # prepare targets
        emb_size = outputs["pred_logits"].shape[2]
        targets = prepare_targets(anno_b, captions, shapes, emb_size)

        loss_dict = criterion(outputs, targets, mask_bi)
        weight_dict = criterion.weight_dict

        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss.backward()
        optimizer.step()
        
        counter_for_image += 1
        results = threshold(outputs, captions, model.tokenizer, TEXT_TRESHOLD)
        for b in range(len(results)): # (bs*num_cap)
            boxes, logits, phrases = results[b]
            boxes = [box.tolist() for box in boxes]
            logits = logits.tolist()

            points = [[box[0], box[1]] for box in boxes] # center points

            # calculate error
            pred_cnt = len(points)
            gt_cnt = len(targets[b]["points"])
            cnt_err = abs(pred_cnt - gt_cnt)
            train_mae += cnt_err
            train_rmse += cnt_err ** 2

            # calculate loc metric
            TP, FP, FN, precision, recall, f1 = calc_loc_metric(boxes, targets[b]["points"])
            train_tp += TP
            train_fp += FP
            train_fn += FN
        
            counter += 1
            
            print(f'[train] ep {epoch} ({counter_for_image}/{train_size}), {img_caps[b]}, caption: {captions[b]}, actual-predicted: {gt_cnt} vs {pred_cnt}, error: {pred_cnt - gt_cnt}. Current MAE: {int(train_mae/counter)}, RMSE: {int((train_rmse/counter)**0.5)} | TP = {TP}, FP = {FP}, FN = {FN}, precision = {precision:.2f}, recall = {recall:.2f}, F1 = {f1:.2f}')
        
        
    
    train_mae = train_mae / counter
    train_rmse = (train_rmse / counter) ** 0.5

    train_precision = train_tp / (train_tp + train_fp) if train_tp + train_fp != 0 else 0.0
    train_recall = train_tp / (train_tp + train_fn) if train_tp + train_fn != 0 else 0.0
    train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall) if train_precision + train_recall != 0 else 0.0

    return train_mae, train_rmse, train_tp, train_fp, train_fn, train_precision, train_recall, train_f1




def eval(split, epoch=None):
    print(f"Evaluation on {split} set")
    model.eval()
    loader = loaders[split]

    eval_mae = 0
    eval_rmse = 0

    eval_tp = 0
    eval_fp = 0
    eval_fn = 0
    
    counter = 0
    counter_for_image = 0
    eval_size = len(loader.dataset)

    for images, captions, shapes, img_caps in loader: # tensor, list, list, list

        anno_b = [annotations[img_cap] for img_cap_list in img_caps for img_cap in img_cap_list] 
        img_caps = [img_cap for img_cap_list in img_caps for img_cap in img_cap_list]
        shapes = [shapes[i] for i, caption_list in enumerate(captions) for _ in caption_list]

        
        images = torch.stack([images[i] for i, caption_list in enumerate(captions) for _ in caption_list], dim=0)
        captions = [caption for caption_list in captions for caption in caption_list] # flatten list of list
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images, captions=captions)

        outputs["pred_points"] = outputs["pred_boxes"][:, :, :2] 

        # prepare targets
        emb_size = outputs["pred_logits"].shape[2]
        targets = prepare_targets(anno_b, captions, shapes, emb_size)


        counter_for_image += 1
        
        results = threshold(outputs, captions, model.tokenizer, TEXT_TRESHOLD)
        for b in range(len(results)):
            boxes, logits, phrases = results[b]
            boxes = [box.tolist() for box in boxes]
            logits = logits.tolist()


            points = [[box[0], box[1]] for box in boxes]

            # calculate error
            pred_cnt = len(points)
            gt_cnt = len(targets[b]["points"])
            cnt_err = abs(pred_cnt - gt_cnt)
            eval_mae += cnt_err
            eval_rmse += cnt_err ** 2
        
            # calculate loc metric
            TP, FP, FN, precision, recall, f1 = calc_loc_metric(boxes, targets[b]["points"])
            eval_tp += TP
            eval_fp += FP
            eval_fn += FN


            counter += 1
            
            print(f'[{split}] ep {epoch} ({counter_for_image}/{eval_size}), {img_caps[b]}, caption: {captions[b]}, actual-predicted: {gt_cnt} vs {pred_cnt}, error: {pred_cnt - gt_cnt}. Current MAE: {int(eval_mae/counter)}, RMSE: {int((eval_rmse/counter)**0.5)} | TP = {TP}, FP = {FP}, FN = {FN}, precision = {precision:.2f}, recall = {recall:.2f}, F1 = {f1:.2f}')
        
        

    eval_mae = eval_mae / counter
    eval_rmse = (eval_rmse / counter) ** 0.5

    eval_precision = eval_tp / (eval_tp + eval_fp) if eval_tp + eval_fp != 0 else 0.0
    eval_recall = eval_tp / (eval_tp + eval_fn) if eval_tp + eval_fn != 0 else 0.0
    eval_f1 = 2 * eval_precision * eval_recall / (eval_precision + eval_recall) if eval_precision + eval_recall != 0 else 0.0

    return eval_mae, eval_rmse, eval_tp, eval_fp, eval_fn, eval_precision, eval_recall, eval_f1


def prepare_targets(anno_b, captions, shapes, emb_size):
    for anno in anno_b:
        if len(anno['points']) == 0:
            anno['points'] = [[0,0]]
    gt_points_b = [np.array(anno['points']) / np.array(shape)[::-1] for anno, shape in zip(anno_b, shapes)] # (h,w) -> (w,h)
    gt_points = [torch.from_numpy(img_points).to(torch.float32) for img_points in gt_points_b] 

    gt_logits = [torch.zeros((img_points.shape[0], emb_size)) for img_points in gt_points] 

    
    tokenized = model.tokenizer(captions, padding="longest", return_tensors="pt")

    # find last index of special token (.)
    end_idxes = [torch.where(input_ids==1012)[0][-1] for input_ids in tokenized['input_ids']] 
    for i, end_idx in enumerate(end_idxes):
        gt_logits[i][:,:end_idx] = 1.0 

    caption_sizes = [end_idx + 2 for end_idx in end_idxes]  # incl. CLS and SEP

    targets = [{"points": img_gt_points.to(device), "labels": img_gt_logits.to(device), "caption_size": caption_size} for img_gt_points, img_gt_logits, caption_size in zip(gt_points, gt_logits, caption_sizes)] 

    return targets


def distance_threshold_func(boxes): # list of [xc,yc,w,h]
    if len(boxes) == 0:
        return 0.0
    # find median index of boxes areas
    areas = [box[2]*box[3] for box in boxes]
    median_idx = np.argsort(areas)[len(areas)//2]
    median_box = boxes[median_idx]
    w = median_box[2]
    h = median_box[3]

    threshold = np.sqrt(w**2 + h**2) / 2.0
    
    return threshold

def calc_loc_metric(pred_boxes, gt_points): # list of [xc,yc,w,h], tensor of (nt,2)
    if len(pred_boxes) == 0:
        FN = len(gt_points)
        return 0, 0, FN, 0, 0, 0
    
    dist_threshold = distance_threshold_func(pred_boxes)
    pred_points = np.array([[box[0], box[1]] for box in pred_boxes])
    gt_points = gt_points.cpu().detach().numpy()

    # create a cost matrix
    cost_matrix = cdist(pred_points, gt_points, metric='euclidean')
    
    # use Hungarian algorithm to find optimal assignment
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # determine TP, FP, FN
    TP = 0
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if cost_matrix[pred_idx, gt_idx] < dist_threshold:
            TP += 1
    
    FP = len(pred_points) - TP
    FN = len(gt_points) - TP

    Precision = TP / (TP + FP) if TP + FP != 0 else 0.0
    Recall = TP / (TP + FN) if TP + FN != 0 else 0.0
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if Precision + Recall != 0 else 0.0
    
    return TP, FP, FN, Precision, Recall, F1


# main 

stats_dir = "./stats"
os.makedirs(stats_dir, exist_ok=True)

stats_file = f"{stats_dir}/stats.txt"
stats = list()

print(f"Saving stats to {stats_file}")

with open(stats_file, 'a') as f:
    header = ['train_mae', 'train_rmse', 'train_TP', 'train_FP', 'train_FN', 'train_precision', 'train_recall', 'train_f1', '||', 'val_mae', 'val_rmse', 'val_TP', 'val_FP', 'val_FN', 'val_precision', 'val_recall', 'val_f1', '||', 'test_mae', 'test_rmse', 'test_TP', 'test_FP', 'test_FN', 'test_precision', 'test_recall', 'test_f1']
    f.write("%s\n" % ' | '.join(header))


best_f1 = 0.0
best_model = None
for epoch in range(0, 15):

    train_mae, train_rmse, train_TP, train_FP, train_FN, train_precision, train_recall, train_f1 = train(epoch)
    val_mae, val_rmse, val_TP, val_FP, val_FN, val_precision, val_recall, val_f1 = eval('val', epoch)
    
    if best_f1 < val_f1:
        best_f1 = val_f1
        print(f"New best F1: {best_f1}")
        best_model = copy.deepcopy(model)
    
    stats.append([train_mae, train_rmse, train_TP, train_FP, train_FN, train_precision, train_recall, train_f1, "||", val_mae, val_rmse, val_TP, val_FP, val_FN, val_precision, val_recall, val_f1, "||", 0,0,0,0,0, 0,0,0])

    with open(stats_file, 'a') as f:
        s = stats[-1]
        for i, x in enumerate(s):
            if type(x) != str:
                s[i] = str(round(x,4))
        f.write("%s\n" % ' | '.join(s))

model_name = f'{stats_dir}/model.pth'
torch.save({"model": best_model.state_dict()}, model_name)


# Inference on test set
print(f"Inference on test set using best model: {model_name}")
model = load_model(CONFIG_PATH, model_name)
model = model.to(device)
test_mae, test_rmse, test_TP, test_FP, test_FN, test_precision, test_recall, test_f1 = eval('test', -1)
print(f"test MAE: {test_mae:5.2f}, RMSE: {test_rmse:5.2f}, TP: {test_TP}, FP: {test_FP}, FN: {test_FN}, precision: {test_precision:5.2f}, recall: {test_recall:5.2f}, f1: {test_f1:5.2f}")
# write to stats file
line_inference = [0,0,0,0,0, 0,0,0, "||", 0,0,0,0,0, 0,0,0, "||", test_mae, test_rmse, test_TP, test_FP, test_FN, test_precision, test_recall, test_f1]
with open(stats_file, 'a') as f:
    s = line_inference
    for i, x in enumerate(s):
        if type(x) != str:
            s[i] = str(round(x,4))
    f.write("%s\n" % ' | '.join(s))


