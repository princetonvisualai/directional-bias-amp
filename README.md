# Metric
Code to replicate [Directional Bias Amplification](https://arxiv.org/abs/2102.12594)

To measure Directional Bias Amplification, use the metric code in directional_biasamp.py

# Replicate Experiments
Each of the 7 folders contains the code needed to replicate the experiments: 
- paint_ex (painting example that creates Figure 2)
- coco_captioning (investigates equalizer model for Figure 3 examples)
- coco_mask (experiment with masking in COCO in Table 1)
- mals (men also like shopping with thresholding in Section 3.3)
- celeba (variance in estimator bias experiments for Section 5.2 and Figure 4)
- fit_bert (section 5.3 "Sentence completion: no ground truth.")
- compas (section 5.3 "Risk prediction: future outcomes unknown.")

Each folder contains a file, run.sh, that contains the necessary scripts to run.

# Python Packages
- python = 3.7.3
- numpy = 1.19.1
- torch = 1.4.0
- torchvision = 0.5.0
- responsibly = 0.1.1
- scipy = 1.4.1
