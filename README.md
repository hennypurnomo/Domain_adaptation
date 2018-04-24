# Domain_adaptation

This project is about Domain Adaptation with Evolutionary Strategies on several datasets, which are MNIST-M and Office-31.

This project build the classifier with combination of Neural Network and two Random Forests. Also, utilize SNES algorithm for Evolutionary Strategies. Neural network is used to extract the features from data, and it is continued with 2 different tasks on Random Forests. First Random Forest has to classify the label of class, example on MNIST-M data, the label is 0 until 9, where as the second Random Forest has to learn to mapping the features from both data.

To get the data from MNIST-M, you can download it from https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500. While for Office-31, you can download from https://pan.baidu.com/s/1o8igXT4#list/path=%2F. 

To run this project, it should be started from :
 1. Run the create_mnistm.py, to create a pickle on MNIST-M data (It can not be included on this Github because limit of size to upload the file)
 2. Compile MNIST_MNISTM.ipynb to run testing on MNIST-M data.
 3. To run each experiment on Office-31, choose one of these projects: amazon_webcam.ipynb, dslr_webcam.ipynb, amazon_dslr.ipynb.
