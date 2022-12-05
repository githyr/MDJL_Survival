# MDJL_Survival
This is an implementation of Joint Learning Sample Similarity and Correlation Representation for Cancer Survival prediction (MDJL) in Pytorch.
## Requirements
  * Python=3.5.6  
  * Pytorch=1.0.0  
  * Torchvision=0.2.1  
## Implementation
Training MDJL can be done in a few lines. First, all you need to do is prepare the datasets to have the following keys:
    
    {  
        'X': (n,d) observations (dtype = float32),  
        'y': (n) event times (dtype = float32),  
        'e': (n) event indicators (dtype = int32)
    }      
 

You can then evaluate its success on survival data with training set and test set:  
`` python Main.py ``
## Citation
Joint Learning Sample Similarity and Correlation Representation for Cancer Survival prediction, BMC Bioinformatics.
