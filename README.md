# Deep Learning for Plant Disease Detection

This project showcases an experiment described in the chapter "Deep Learning for Plant Diseases: Detection and Saliency Map Visualisation" in the book "Human and Machine Learning."

The project involves training and evaluating state-of-the-art deep architectures for plant disease classification using pyTorch. The dataset consists of 38 disease classes from the PlantVillage dataset and one background class from Stanford's open dataset of background images (DAGS). The models are trained on a preprocessed dataset.


1. Train all the models and store the evaluation stats in **stats.csv**:
   `python3 train.py`

2. Plot the models' results for every architecture based on the stored stats with **plot.py**:
   `python3 plot.py`

## Results:
The models on the graph were retrained on final fully connected layers only (shallow), for the entire set of parameters (deep), or from their initialized state (from scratch).

## Prerequisites:
Train the new model or download pretrained models on **10 classes** of **Tomato** from the PlantVillage dataset: [AlexNet](https://drive.google.com/open?id=1Ms1Ri5DUy_D4uYZX5tG2hrN2hUH6XbQS) or [VGG13](https://drive.google.com/open?id=1f0nPNRfL42fJA8tF5JoKUKv0Xr98p8-P).

## Occlusion Experiment
The occlusion experiment produces heat maps that visually show the influence of each region on the classification.

### Usage:
Produce the heat map and plot with **occlusion.py** and store the visualizations in **output_dir**:

`python3 occlusion.py /path/to/dataset /path/to/output_dir model_name.pkl /path/to/image disease_name`

### Visualization Examples on AlexNet:
![Early Blight](https://raw.githubusercontent.com/MarkoArsenovic/DeepLearning_PlantDiseases/master/Scripts/visualization/output/early_blight/early_blight.png)
*Early blight - original, size 80 stride 10, size 100 stride 10*

![Late Blight](https://raw.githubusercontent.com/MarkoArsenovic/DeepLearning_PlantDiseases/master/Scripts/visualization/output/late_blight/late_blight.png)
*Late blight - original, size 80 stride 10, size 100 stride 10*

![Septoria Leaf Spot](https://raw.githubusercontent.com/MarkoArsenovic/DeepLearning_PlantDiseases/master/Scripts/visualization/output/septoria_leaf_spot/septoria_leaf_spot.png)
*Septoria leaf spot - original, size 50 stride 10, size 100 stride 10*

## Saliency Map Experiment
The saliency map is an analytical method that estimates the importance of each pixel using only one forward and one backward pass through the network.

### Usage:
Produce the visualization and plot with **saliency.py** and store the visualizations in **output_dir**:

`python3 occlusion.py /path/to/model /path/to/dataset /path/to/image class_name`

