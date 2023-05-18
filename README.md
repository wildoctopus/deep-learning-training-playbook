# Effective Deep Learning Training Strategies
Useful Optimized and Effective Deep Learning training strategies for improved model performance



## Table of Contents

-   [Who can benefit from this document?](#Who-can-benefit-from-this-document?)
-   [Why this Effective Training Strategies?](#Why-this-Effective-Training-Strategies?)
-   [Classification Models](#Classification-Models)
    - [Image Classification](#Image-Classification)
-   [Generative Models](#Generative-Models)
    - [Training GAN](#Training-GAN)


## Who is this document for?

This guide targets both individual engineers and research teams who are interested in optimizing the performance of deep learning models. It assumes a basic understanding of machine learning and deep learning concepts.

We acknowledge that there is no universal solution that fits all scenarios. Therefore, our main focus is on offering strategies that can be commonly followed, taking into account the problem type or the characteristics of the dataset being used. These insights have been derived from experiments conducted on different categorizations. 

Our current emphasis is not on hyperparameter tuning; instead, we concentrate on a generic pipeline with parameters based on observations from personal experience.

## Why this Effective Training Strategies?

The field of deep learning currently involves a significant amount of trial and error, lacking a systematic approach for achieving optimal results. Documentation of successful strategies is scarce, with research papers often omitting crucial details and practical insights. Machine learning engineers working on real-world problems seldom have the opportunity to generalize their processes. Existing resources, such as textbooks, prioritize theoretical concepts over practical guidance. Despite the expertise of deep learning practitioners, there remains a gap between their results and those of less experienced individuals. The absence of comprehensive explanations and the prevalence of fragmented advice contribute to confusion in the community. As deep learning continues to mature and impact various domains, there is a pressing need for resources that provide practical recipes and address the essential details for achieving desirable outcomes.


## Classification Models

### Image Classification

Well a common strategy regardless of dataset charactersitics for this task is to apply Transfer Learning , choosing a pretrained model, add a custom head layer as per requirement and then train that model with several hyperparameter configuration. But we will discus strategies which are more generic and preloaded with already configured hyperparamters for specific choice of model.

Now these strategies are divided based on the Daataset Charactersisics and the choice of model. In general, considering the similarity of custom dataset with pre-trained model dataset, a problem can fall into one of these quadrants (as shown in image). 
- Common to every strategy : 
    - Data Preprocessing : Normalize Images using standard values -
    - Data splitting and Smapling : Stratified Splitting and Weighted Sampling. This helps better generalization by including samples based on clas weights or Inverse clas frequenccey.
- Strategy 1 :  In case when classification problem falls in Quadrant 1, choose base model based on its complexity. If you dataset contains complex pattern, choose either from EfficientNet or DenseNet family. Use stratified splitting and Weighted Sampling while creating dataloaders, so that sampling is performed based on class weights. It helps in better generaliation. To decide batch size.
    - Data  
    - Batch size : 
    - Learning rate : 
    - Learning rate scheduler : 
- Strategy 2 :  In case when classification problem falls in Quadrant 1, choose base model based on its complexity. If you dataset contains complex pattern, choose either from EfficientNet or DenseNet family. Use stratified splitting and Weighted Sampling while creating dataloaders, so that sampling is performed based on class weights. It helps in better generaliation. To decide batch size.
    - Data  
    - Batch size : 
    - Learning rate : 
    - Learning rate scheduler :
- Strategy 3 :  In case when classification problem falls in Quadrant 1, choose base model based on its complexity. If you dataset contains complex pattern, choose either from EfficientNet or DenseNet family. Use stratified splitting and Weighted Sampling while creating dataloaders, so that sampling is performed based on class weights. It helps in better generaliation. To decide batch size.
    - Data  
    - Batch size : 
    - Learning rate : 
    - Learning rate scheduler :
- Strategy 4 :  In case when classification problem falls in Quadrant 1, choose base model based on its complexity. If you dataset contains complex pattern, choose either from EfficientNet or DenseNet family. Use stratified splitting and Weighted Sampling while creating dataloaders, so that sampling is performed based on class weights. It helps in better generaliation. To decide batch size.
    - Data  
    - Batch size : 
    - Learning rate : 
    - Learning rate scheduler :


## Generative Models

### Training GAN
