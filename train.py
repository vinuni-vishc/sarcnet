import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import argparse
import time
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from utils import load_image, process_csv, process_meta
from models import SarcNet

parser = argparse.ArgumentParser()
parser.add_argument('--csvpath1', type=str, required=True)
parser.add_argument('--csvpath2', type=str, required=True)
parser.add_argument('--metapath1', type=str, required=True)
parser.add_argument('--metapath2', type=str, required=True)
parser.add_argument('--datapath1', type=str, required=True)
parser.add_argument('--datapath2', type=str, required=True)
parser.add_argument('--pixel_size_xy_in_micrometers', type=float, default=0.12)
parser.add_argument('--cuda', type=int,default=0)
parser.add_argument('--numworkers', type=int, default=3)
parser.add_argument('--outpath', type=str, default='./')
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=40)

args = parser.parse_args()

NUM_WORKERS = args.numworkers

if args.cuda >= 0:
    DEVICE = torch.device("cuda:%d" % args.cuda)
else:
    DEVICE = torch.device("cpu")

PATH = args.outpath
if not os.path.exists(PATH):
    os.mkdir(PATH)
LOGFILE = os.path.join(PATH, 'training.log')
TEST_PREDICTIONS = os.path.join(PATH, 'test_predictions.log')
TEST_TRUES = os.path.join(PATH, 'test_trues.log')
TEST_BOTH =  os.path.join(PATH, 'test_both.log')

# Architecture
BATCH_SIZE = args.batch_size

all_fish_df = process_csv(args.csvpath1, args.csvpath2)
all_gs_df = process_meta(args.metapath1, args.metapath2)

fish_df = pd.merge(
        left=all_fish_df,
        right=all_gs_df,
        on=["napariCell_ObjectNumber", "fov_path"],
        how="inner",
    )

# Clean up columns
feature_columns_in = [
    "napariCell_AreaShape_Area",
    "napariCell_AreaShape_MinorAxisLength",
    "napariCell_AreaShape_MajorAxisLength",
    "Frac_Area_Background",
    "Frac_Area_DiffuseOthers",
    "Frac_Area_Fibers",
    "Frac_Area_Disorganized_Puncta",
    "Frac_Area_Organized_Puncta",
    "Frac_Area_Organized_ZDisks",
    "Maximum_Coefficient_Variation",
    "Peak_Height",
    "Peak_Distance",
]
metadata_cols_in = [
    "napariCell_ObjectNumber",
    "rescaled_2D_single_cell_tiff_path",
    "fov_path",
    "cell_age",
    "mh_structure_org_score",
    "kg_structure_org_score",
]

fish_df = fish_df[metadata_cols_in + feature_columns_in]

fish_df["Cell aspect ratio"] = (fish_df["napariCell_AreaShape_MinorAxisLength"]
/ fish_df["napariCell_AreaShape_MajorAxisLength"]
)
fish_df = fish_df.drop(
    columns=[
        "napariCell_AreaShape_MinorAxisLength",
        "napariCell_AreaShape_MajorAxisLength",
    ]
    )

# fix units on length / area cols
fish_df["napariCell_AreaShape_Area"] = (
    fish_df["napariCell_AreaShape_Area"] * args.pixel_size_xy_in_micrometers ** 2
)
fish_df["Peak_Distance"] = fish_df["Peak_Distance"] * args.pixel_size_xy_in_micrometers

fish_df = fish_df.rename(
    columns={
        "napariCell_AreaShape_Area": "Cell area (μm^2)",
        "Frac_Area_Background": "Fraction cell area background",
        "Frac_Area_DiffuseOthers": "Fraction cell area diffuse/other",
        "Frac_Area_Fibers": "Fraction cell area fibers",
        "Frac_Area_Disorganized_Puncta": "Fraction cell area disorganized puncta",
        "Frac_Area_Organized_Puncta": "Fraction cell area organized puncta",
        "Frac_Area_Organized_ZDisks": "Fraction cell area organized z-disks",
        "Maximum_Coefficient_Variation": "Max coefficient var",
        "Peak_Height": "Peak height",
        "Peak_Distance": "Peak distance (μm)",
        "cell_age": "Cell age",
    }
)

# Calculate mean expert score
fish_df["Expert structural annotation score (mean)"] = fish_df[
    ["mh_structure_org_score", "kg_structure_org_score"]
].mean(axis="columns")

assert len(fish_df) == len(
    fish_df[["napariCell_ObjectNumber", "fov_path"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

##### LINEAR REGRESSION #####

# columns / facet definitons and order
BAR_PLOT_COLUMNS = [
    "rescaled_2D_single_cell_tiff_path",
    "Cell area (μm^2)",
    "Cell aspect ratio",
    "Fraction cell area background",
    "Fraction cell area diffuse/other",
    "Fraction cell area fibers",
    "Fraction cell area disorganized puncta",
    "Fraction cell area organized puncta",
    "Fraction cell area organized z-disks",
    "Max coefficient var",
    "Peak height",
    "Peak distance (μm)",
]

LINEAR_COLUMNS = [
    "Cell area (μm^2)",
    "Cell aspect ratio",
    "Fraction cell area background",
    "Fraction cell area diffuse/other",
    "Fraction cell area fibers",
    "Fraction cell area disorganized puncta",
    "Fraction cell area organized puncta",
    "Fraction cell area organized z-disks",
    "Max coefficient var",
    "Peak height",
    "Peak distance (μm)",
]

### Dataframe and model
all_good_scores = (fish_df.kg_structure_org_score > 0) & (
    fish_df.mh_structure_org_score > 0
)
fish_df = fish_df[all_good_scores].reset_index(drop=True) #(5761,18)

train, test = train_test_split(fish_df, test_size=0.2, random_state=42, stratify=fish_df['mh_structure_org_score'])
train, val = train_test_split(train, test_size=0.2, random_state=42, stratify=train['mh_structure_org_score'])

feat_cols = [c for c in BAR_PLOT_COLUMNS if c in fish_df.columns]

#DataFrame for Machine Learning (Linear Regression)
df_train = fish_df.loc[train.index, feat_cols + ["Expert structural annotation score (mean)"]].copy()
df_test = fish_df.loc[test.index, feat_cols + ["Expert structural annotation score (mean)"]].copy()
df_val = fish_df.loc[val.index, feat_cols + ["Expert structural annotation score (mean)"]].copy()

custom_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, ], [0.5, ])
    ]
)


#Get images
print('---> Load & Transform Images start')

#Apply transformations to training and testing images
transformed_train_X = torch.stack(load_image(df_train, custom_transform, args.datapath1, args.datapath2))
transformed_test_X = torch.stack(load_image(df_test, custom_transform, args.datapath1, args.datapath2))
transformed_val_X = torch.stack(load_image(df_val, custom_transform, args.datapath1, args.datapath2))

print('---> Load & Transform Images done')

print(transformed_train_X.shape) #torch.Size([4608, 3, 224, 224])

scaler = StandardScaler()
scaled_features_train = scaler.fit_transform(df_train[LINEAR_COLUMNS])
scaled_features_test = scaler.transform(df_test[LINEAR_COLUMNS])
scaled_features_val = scaler.transform(df_val[LINEAR_COLUMNS])

features_train_tensor = torch.tensor(scaled_features_train, dtype=torch.float32)
features_test_tensor = torch.tensor(scaled_features_test, dtype=torch.float32)
features_val_tensor = torch.tensor(scaled_features_val, dtype=torch.float32)

print(features_train_tensor.shape)

train_dataset = []
for i in range(df_train.shape[0]):
    # Example tensors for each set
    tensor1 = features_train_tensor[i]
    tensor2 = transformed_train_X[i]
    tensor3 = df_train["Expert structural annotation score (mean)"].tolist()[i] 
    train_dataset.append((tensor1, tensor2, tensor3))

test_dataset = []
for i in range(df_test.shape[0]):
    # Example tensors for each set
    tensor1 = features_test_tensor[i]
    tensor2 = transformed_test_X[i]
    tensor3 = df_test["Expert structural annotation score (mean)"].tolist()[i]
    test_dataset.append((tensor1, tensor2, tensor3))
    
val_dataset = []
for i in range(df_val.shape[0]):
    # Example tensors for each set
    tensor1 = features_val_tensor[i]
    tensor2 = transformed_val_X[i]
    tensor3 = df_val["Expert structural annotation score (mean)"].tolist()[i]
    val_dataset.append((tensor1, tensor2, tensor3))

#DataLoader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS,
                          )

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS)

val_loader = DataLoader(dataset=val_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS)

print('---> DataLoader done')

# ##########################
# # MODEL
# ##########################

# Loss function
loss_fn = nn.MSELoss()

model = SarcNet(num_features=len(LINEAR_COLUMNS))
model.to(DEVICE)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

start_time = time.time()

print('---> Training Start')
best_mse, best_epoch, best_corr, best_epoch2 = 999, -1, -1, -1
loss_train = []
loss_test = []
for epoch in range(args.num_epochs):
    losses = []
    model.train()
    for batch_idx, (features, images, targets) in enumerate(train_loader):
        features = features.to(DEVICE)
        images = images.to(DEVICE)

        targets = targets.float().to(DEVICE)

        # FORWARD AND BACK PROP
        output = model(features, images)

        loss = loss_fn(torch.squeeze(output), targets)

        losses.append(loss.item())

        optimizer.zero_grad()

        loss.backward()

        # UPDATE MODEL PARAMETERS
        optimizer.step()

        # LOGGING
        if not batch_idx % 20:
            s = ('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                 % (epoch+1, args.num_epochs, batch_idx,
                     len(train_dataset)//BATCH_SIZE, loss))
            print(s)
            with open(LOGFILE, 'a') as f:
                f.write('%s\n' % s)
    loss_train.append(np.mean(losses))
    
    model.eval()
    all_true_test = []
    all_pred_test = []
    with torch.set_grad_enabled(False):
        num_examples = 0
        test_loss_sum = 0
        for batch_idx, (features, images, targets) in enumerate(val_loader):       
            test_loss_sum = 0
            features = features.to(DEVICE)
            images = images.to(DEVICE)

            targets = targets.to(DEVICE)
            num_examples += targets.size(0)

            # FORWARD AND BACK PROP
            output = model(features, images)
            test_loss = loss_fn(output.squeeze(), targets)

            losses.append(test_loss.item())

            test_loss_sum += test_loss.item()

            output = output.squeeze()

            all_pred_test.extend(output.tolist())

            lst_true = [str(float(i)) for i in targets]
            all_true_test.extend(lst_true)
        
        print(test_loss_sum)
        average_test_loss = test_loss_sum / len(test_loader)
        loss_test.append(average_test_loss)


        all_pred_float = [float(pred) for pred in all_pred_test]  # Assuming each prediction is a tensor with one element
        all_true_float = [float(true) for true in all_true_test]

        # Calculate Spearman correlation
        spearman_corr, _ = spearmanr(all_true_float, all_pred_float)

        if average_test_loss < best_mse:
            best_mse, best_epoch = average_test_loss, epoch
            torch.save(model.state_dict(), os.path.join(PATH, 'best_model_loss.pt'))

        if spearman_corr > best_corr:
            best_corr, best_epoch2 = spearman_corr, epoch
            torch.save(model.state_dict(), os.path.join(PATH, 'best_model_corr.pt'))
  
        s = 'MAE/RMSE: | Current Valid: %.3f| Corr: %.3f Ep. %d | Best Valid : %.3f Ep. %d | Best Corr : %.3f Ep. %d' % (
            average_test_loss, spearman_corr, epoch, best_mse, best_epoch, best_corr, best_epoch2)
        print(s)
    
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

    s = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)
        
    plt.plot(np.array(loss_train), 'b')
    plt.plot(np.array(loss_test), 'orange')
    plt.savefig(os.path.join(PATH, 'loss.png'))

########## EVALUATE BEST MODEL ######
print('MODEL WITH BEST LOSS')
model.load_state_dict(torch.load(os.path.join(PATH, 'best_model_loss.pt')))
model.eval()

########## SAVE PREDICTIONS ######
all_true = []
all_pred = []
with torch.set_grad_enabled(False):
    for batch_idx, (features, images, targets) in enumerate(test_loader):
        features = features.to(DEVICE)
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        output = model(features, images)
        
        output = output.squeeze()
        all_pred.extend(output.tolist())

        lst_true = [str(float(i)) for i in targets]
        all_true.extend(lst_true)

# Convert predictions and true values to lists of floats
all_pred_float = np.array([float(pred) for pred in all_pred])  # Assuming each prediction is a tensor with one element
all_true_float = np.array([float(true) for true in all_true])

# Calculate Spearman correlation
spearman_corr, _ = spearmanr(all_true_float, all_pred_float)
mae = np.sum(np.abs(all_true_float - all_pred_float)) / len(all_pred_float)
r2 = r2_score(all_true_float, all_pred_float)
mse = np.sum((all_true_float - all_pred_float) ** 2) / len(all_pred_float)

print(f'Spearman Correlation: {spearman_corr:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'R2 score: {r2:.4f}')

########## EVALUATE BEST MODEL ######
print('MODEL WITH BEST CORRELATION')
model.load_state_dict(torch.load(os.path.join(PATH, 'best_model_corr.pt')))
model.eval()

########## SAVE PREDICTIONS ######
all_true = []
all_pred = []
with torch.set_grad_enabled(False):
    for batch_idx, (features, images, targets) in enumerate(test_loader):
        features = features.to(DEVICE)
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        output = model(features, images)
        
        output = output.squeeze()
        all_pred.extend(output.tolist())

        lst_true = [str(float(i)) for i in targets]
        all_true.extend(lst_true)

# Convert predictions and true values to lists of floats
all_pred_float = np.array([float(pred) for pred in all_pred])  # Assuming each prediction is a tensor with one element
all_true_float = np.array([float(true) for true in all_true])

# Calculate Spearman correlation
spearman_corr, _ = spearmanr(all_true_float, all_pred_float)
mae = np.sum(np.abs(all_true_float - all_pred_float)) / len(all_pred_float)
r2 = r2_score(all_true_float, all_pred_float)
mse = np.sum((all_true_float - all_pred_float) ** 2) / len(all_pred_float)

print(f'Spearman Correlation: {spearman_corr:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MSE: {mse:.4f}')
print(f'R2 score: {r2:.4f}')
