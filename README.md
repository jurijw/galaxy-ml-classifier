# Galaxy ML Classifier
This project aims to train a machine learning classifier that categorizes images of galaxies into elliptical or spiral morphologies. It does so by using data from the Galaxy Zoo survey to select, and subsequently clean, a subset of SDSS objects. These objects are then requested via the hips2fits api, labeled, and used as training inputs to a convolutional neural network.
![Example elliptical galaxy](examples/elliptical.jpg)
![Feature maps for first convolution layer on example elliptical galaxy](examples/feature_maps_elliptical_layer1.png)
![Feature maps for second convolution layer on example elliptical galaxy](examples/feature_maps_elliptical_layer2.png)
![Example elliptical galaxy](examples/spiral.jpg)
![Feature maps for first convolution layer on example spiral galaxy](examples/feature_maps_spiral_layer1.png)
![Feature maps for second convolution layer on example spiral galaxy](examples/feature_maps_spiral_layer2.png)
