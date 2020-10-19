#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from typing import Dict

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50
from tqdm import tqdm
from torch import Tensor
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path
import csv
import os
import torch.nn.functional as F
import sys
np.set_printoptions( threshold=sys.maxsize)
def pytorch_neg_multi_log_likelihood_batch(gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor) -> Tensor:
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce
    return torch.mean(error)
def pytorch_neg_multi_log_likelihood_single(gt: Tensor, pred: Tensor, avails: Tensor) -> Tensor:
  batch_size, future_len, num_coords = pred.shape
  confidences = pred.new_ones((batch_size, 1))
  return pytorch_neg_multi_log_likelihood_batch(gt, pred.unsqueeze(1), confidences, avails)
class LyftMultiModel(nn.Module):

    def __init__(self, cfg: Dict, num_modes=3):
        super().__init__()

        architecture = cfg["model_params"]["model_architecture"]
        backbone = eval(architecture)(pretrained=True, progress=True)
        self.backbone = backbone

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )
        self.backbone.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
        backbone_out_features = 2048
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len
          
        # You can add more layers here.
        self.head = nn.Sequential(
             nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )

        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.head(x)
        x = self.logit(x)

        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences

    
    
    


# In[ ]:


def forwardm(data, model, device, criterion = pytorch_neg_multi_log_likelihood_batch):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device)
    # Forward pass
    preds, confidences = model(inputs)
    loss = criterion(targets, preds, confidences, target_availabilities)
    return loss, preds, confidences
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    targets = data["target_positions"].to(device)
    # Forward pass
    outputs = model(inputs).reshape(targets.shape)
    loss = criterion(outputs, targets)
    # not all the output steps are valid, but we can filter them out from the loss using availabilities
    loss = loss * target_availabilities
    loss = loss.mean()
    return loss, outputs


# ## Load the Train Data
# 
# Our data pipeline map a raw `.zarr` folder into a multi-processing instance ready for training by:
# - loading the `zarr` into a `ChunkedDataset` object. This object has a reference to the different arrays into the zarr (e.g. agents and traffic lights);
# - wrapping the `ChunkedDataset` into an `AgentDataset`, which inherits from torch `Dataset` class;
# - passing the `AgentDataset` into a torch `DataLoader`

# In[ ]:
# ## Prepare Data path and load cfg
# 
# By setting the `L5KIT_DATA_FOLDER` variable, we can point the script to the folder where the data lies.
# 
# Then, we load our config file with relative paths and other configurations (rasteriser, training params...).

# In[ ]:

if __name__ == '__main__':
# set env variable for data
 os.environ["L5KIT_DATA_FOLDER"] = r'C:\Users\user\Desktop\working\lyft motion prediction\data'
 dm = LocalDataManager(None)
# get config
 cfg = load_config_data("./agent_motion_config.yaml")
 print(cfg)


# ## Model
# 
# Our baseline is a simple `resnet50` pretrained on `imagenet`. We must replace the input and the final layer to address our requirements.

# In[ ]:

# ===== INIT DATASET
 train_cfg = cfg["train_data_loader"]
 rasterizer = build_rasterizer(cfg, dm)
 train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
 train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
 train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"])
 print(train_dataset)


# In[ ]:


# ==== INIT MODEL
 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 model = LyftMultiModel(cfg).to(device)
 optimizer = optim.Adam(model.parameters(), lr=1e-4)
 criterion =pytorch_neg_multi_log_likelihood_batch #nn.MSELoss(reduction="none")


# # Training
# 
# note: if you're on MacOS and using `py_satellite` rasterizer, you may need to disable opencv multiprocessing by adding:
# `cv2.setNumThreads(0)` before the following cell. This seems to only affect running in python notebook and it's caused by the `cv2.warpaffine` function

# In[ ]:


# ==== TRAIN LOOP
 tr_it = iter(train_dataloader)
 progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
 losses_train = []
 for Index1 in progress_bar:
    try:
        data = next(tr_it)
        
        
        #with open('{}.csv'.format(Index1), 'w') as f:
         #for key in data.keys():
           #f.write("%s,%s\n"%(key,data[key]))
    except StopIteration:
        tr_it = iter(train_dataloader)
        data = next(tr_it)
        #with open('{}.csv'.format(Index1), 'w') as f:
         #for key in data.keys():
           #f.write("%s,%s\n"%(key,data[key]))
        #result= pd.DataFrame.from_dict(data,orient='index')
        
              
        #result.to_csv('{}.csv'.format(Index1), encoding='utf-8', index=False)
    model.train()
    torch.set_grad_enabled(True)
    loss, _, _= forwardm(data, model, device)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    #torch.nn.utils.clip_grad_norm(model.parameter(),1)
    optimizer.step()

    losses_train.append(loss.item())
    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")


# ### Plot Loss Curve
# We can plot the train loss against the iterations (batch-wise)

# In[ ]:


 plt.plot(np.arange(len(losses_train)), losses_train, label="train loss")
 plt.legend()
 #plt.show()


# # Evaluation
# 
# Evaluation follows a slightly different protocol than training. When working with time series, we must be absolutely sure to avoid leaking the future in the data.
# 
# If we followed the same protocol of training, one could just read ahead in the `.zarr` and forge a perfect solution at run-time, even for a private test set.
# 
# As such, **the private test set for the competition has been "chopped" using the `chop_dataset` function**.

# In[ ]:


# ===== GENERATE AND LOAD CHOPPED DATASET
 #num_frames_to_chop = 100
 eval_cfg = cfg["val_data_loader"]
# eval_base_path=r'C:\Users\user\Desktop\working\lyft motion prediction\data\scenes\validate_chopped_100'
 #eval_base_path = create_chopped_dataset(dm.require(eval_cfg["key"]), cfg["raster_params"]["filter_agents_threshold"], 
                              #num_frames_to_chop, cfg["model_params"]["future_num_frames"], MIN_FUTURE_STEPS)


# The result is that **each scene has been reduced to only 100 frames**, and **only valid agents in the 100th frame will be used to compute the metrics**. Because following frames in the scene have been chopped off, we can't just look ahead to get the future of those agents.
# 
# In this example, we simulate this pipeline by running `chop_dataset` on the validation set. The function stores:
# - a new chopped `.zarr` dataset, in which each scene has only the first 100 frames;
# - a numpy mask array where only valid agents in the 100th frame are True;
# - a ground-truth file with the future coordinates of those agents;
# 
# Please note how the total number of frames is now equal to the number of scenes multipled by `num_frames_to_chop`. 
# 
# The remaining frames in the scene have been sucessfully chopped off from the data

# In[ ]:

 eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
 #eval_zarr_path = str(Path(eval_base_path) / Path(dm.require(eval_cfg["key"])).name)
 eval_mask_path =r"C:\Users\user\Desktop\working\lyft motion prediction\data\scenes\mask.npz"
 #eval_gt_path = str(Path(eval_base_path) / "gt.csv")

 #eval_zarr = ChunkedDataset(eval_zarr_path).open()
 eval_mask = np.load(eval_mask_path)["arr_0"]
# ===== INIT DATASET AND LOAD MASK
 eval_dataset = AgentDataset(cfg, eval_zarr, rasterizer, agents_mask=eval_mask)
 eval_dataloader = DataLoader(eval_dataset, shuffle=eval_cfg["shuffle"], batch_size=eval_cfg["batch_size"], 
                             num_workers=eval_cfg["num_workers"])
 print(eval_dataset)


# ### Storing Predictions
# There is a small catch to be aware of when saving the model predictions. The output of the models are coordinates in `agent` space and we need to convert them into displacements in `world` space.
# 
# To do so, we first convert them back into the `world` space and we then subtract the centroid coordinates.

# In[ ]:


# ==== EVAL LOOP
 model.eval()
 torch.set_grad_enabled(False)

# store information for evaluation
 future_coords_offsets_pd = []
 timestamps = []
 agent_ids = []
 confidences_list = []
 progress_bar = tqdm(eval_dataloader)
 for data in progress_bar:
    _, preds, confidences  = forwardm(data, model, device)
    
     #fix for the new environment
    preds = preds.cpu().numpy()
    world_from_agents = data["world_from_agent"].numpy()
    centroids = data["centroid"].numpy()
    coords_offset = []
    for idx in range(len(preds)):
            for mode in range(3):
                preds[idx, mode, :, :] = transform_points(preds[idx, mode, :, :], world_from_agents[idx]) - centroids[idx][:2]
    
    future_coords_offsets_pd.append(preds.copy())
    confidences_list.append(confidences.cpu().numpy().copy())
    timestamps.append(data["timestamp"].numpy().copy())
    agent_ids.append(data["track_id"].numpy().copy())
    


# ### Save results
# After the model has predicted trajectories for our evaluation set, we can save them in a `csv` file.
# 
# During the competition, only the `.zarr` and the mask will be provided for the private test set evaluation.
# Your solution is expected to generate a csv file which will be compared to the ground truth one on a separate server

# In[ ]:


 #pred_path = f"{gettempdir()}/pred.csv"
 pred_path="pred1.csv"
 write_pred_csv(pred_path,
           timestamps=np.concatenate(timestamps),
           track_ids=np.concatenate(agent_ids),
           coords=np.concatenate(future_coords_offsets_pd),
           confs = np.concatenate(confidences_list)
          )


# ### Perform Evaluation
# Pleae note that our metric supports multi-modal predictions (i.e. multiple predictions for a single GT trajectory). In that case, you will need to provide a confidence for each prediction (confidences must all be between 0 and 1 and sum to 1).
# 
# In this simple example we don't generate multiple trajectories, so we won't pass any confidences vector. Internally, the metric computation will assume a single trajectory with confidence equal to 1

# In[ ]:


 metrics = compute_metrics_csv(eval_gt_path, pred_path, [neg_multi_log_likelihood, time_displace])
 for metric_name, metric_mean in metrics.items():
    print('reach here')
    print(metric_name, metric_mean)


# ### Visualise Results
# We can also visualise some results from the ego (AV) point of view for those frames of interest (the 100th of each scene).
# 
# However, as we chopped off the future from the dataset **we must use the GT csv if we want to plot the future trajectories of the agents**
# 

# In[ ]:


 

