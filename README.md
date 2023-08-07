The project involves training and evaluating state-of-the-art deep architectures for plant disease classification using pyTorch. The dataset consists of 38 disease classes from the PlantVillage dataset and one background class from Stanford's open dataset of background images (DAGS). The models are trained on a preprocessed dataset.

Usage:
Train all the models and store the evaluation stats in stats.csv:
python3 train.py

Plot the models' results for every architecture based on the stored stats with plot.py:
python3 plot.py

Results:
The models on the graph were retrained on final fully connected layers only (shallow), for the entire set of parameters (deep), or from their initialized state (from scratch).

Model	Training type	Training time [~h]	Accuracy Top 1
AlexNet	shallow	0.87	0.9415
AlexNet	from scratch	1.05	0.9578
AlexNet	deep	1.05	0.9924
DenseNet169	shallow	1.57	0.9653
DenseNet169	from scratch	3.16	0.9886
DenseNet169	deep	3.16	0.9972
Inception_v3	shallow	3.63	0.9153
Inception_v3	from scratch	5.91	0.9743
Inception_v3	deep	5.64	0.9976
ResNet34	shallow	1.13	0.9475
ResNet34	from scratch	1.88	0.9848
ResNet34	deep	1.88	0.9967
Squeezenet1_1	shallow	0.85	0.9626
Squeezenet1_1	from scratch	1.05	0.9249
Squeezenet1_1	deep	2.10	0.992
VGG13	shallow	1.49	0.9223
VGG13	from scratch	3.55	0.9795
VGG13	deep	3.55	0.9949
NOTE: For more results, refer to stats.csv

Visualization Experiments
Contributor: Brahimi Mohamed

Prerequisites:
Train the new model or download pretrained models on 10 classes of Tomato from the PlantVillage dataset: AlexNet or VGG13.

Occlusion Experiment
The occlusion experiment produces heat maps that visually show the influence of each region on the classification.

Usage:
Produce the heat map and plot with occlusion.py and store the visualizations in output_dir:

python3 occlusion.py /path/to/dataset /path/to/output_dir model_name.pkl /path/to/image disease_name

Visualization Examples on AlexNet:
Early Blight
Early blight - original, size 80 stride 10, size 100 stride 10

Late Blight
Late blight - original, size 80 stride 10, size 100 stride 10

Septoria Leaf Spot
Septoria leaf spot - original, size 50 stride 10, size 100 stride 10

Saliency Map Experiment
The saliency map is an analytical method that estimates the importance of each pixel using only one forward and one backward pass through the network.

Usage:
Produce the visualization and plot with saliency.py and store the visualizations in output_dir:

python3 occlusion.py /path/to/model /path/to/dataset /path/to/image class_name

