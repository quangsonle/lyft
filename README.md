this is for the competition of lyft motion prediction on kaggle https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/submissions

the solution is run on personal computer for some conveniences, only submission file is uploaded here 

this is a testing of drop out based on "https://www.kaggle.com/huanvo/lyft-complete-train-and-prediction-pipeline" 

will be uploaded on github if score is significant enough, otherwise it is a failure

the significant contribution from  the orginal kernel here:

1/ drop out immeidately after the network in resnet50 (injected drop out)

2/ modification of learning rate to deal with explosing gradient in original model

3/ fix the problem of access denied when chop valuate data set using append for zarr array on Window environment

unfortunately, the result is not as well on kaggle. 

However any people get problems with access denied if running on window (your own Desktop instead of kaggle machine, it is better to test something privately) or explosing gradient while training with the above kernel can refer
