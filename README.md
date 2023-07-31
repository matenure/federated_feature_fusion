# federated_feature_fusion
Merging Models Pre-Trained on Different Features with Consensus Graphs - UAI2023

# Usage
Data: please unzip traffic-la.zip and pems-bay.zip and put all data files under the fodler "traffic".

Step 1: train all locla models (this is a one-time step)
```
python main_traffic-la.py --local
```

Step 2: train and test global models
```
python -u main_traffic-la_soft.py --permute
```
or using hard alignment
```
python main_traffic-la.py --permute
```

Additional hyperparameters (such as "epoch","MLP","isGumbel","lr","seed") can be added to change the setting. Please refer to the file "run.sh". 
Note: 
1. As we showed in the appendix of the paper, the soft and hard alignment have simialr performance, and for simplicity we reported the soft alignment results in the main paper. 
2. In addition, versions in "requirement.txt" are not exact requirements, users can choose later-version pytorch (such as 1.9.0) and corresponding packages.
3. PMU data is private, so we only uploaded the public traffic data. The sample code here is used defaultly only for the traffic-la data; for other dataset the model code is the same, and the main code may need small changes to adapt.



# cite
```
@inproceedings{
ma2023federated,
title={Federated Learning of Models Pre-Trained on Different Features with Consensus Graphs},
author={Tengfei Ma and Trong Nghia Hoang and Jie Chen},
booktitle={The 39th Conference on Uncertainty in Artificial Intelligence},
year={2023},
url={https://openreview.net/forum?id=gSMiXJmMEOf}
}
```
