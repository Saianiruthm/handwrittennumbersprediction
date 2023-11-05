# handwrittennumbersprediction

**Handwritten Digit Recognition with TensorFlow**

This project demonstrates how to use TensorFlow to train a neural network to classify handwritten digits from the MNIST dataset. The MNIST dataset is a collection of 60,000 training images and 10,000 testing images of handwritten digits, each of which is 28 pixels by 28 pixels.

**Prerequisites**

To run this project, you will need to install the following prerequisites:

* Python 3.6 or higher
* TensorFlow 2.6 or higher
* NumPy

You can install these prerequisites using the following commands:

```
pip install python
pip install tensorflow
pip install numpy
```

**Instructions**

1. Download the MNIST dataset from the following link: https://www.kaggle.com/code/prashant111/mnist-deep-neural-network-with-keras
2. Extract the downloaded files to a directory on your computer.
3. Save the `mnist_digit_recognizer.py` script to the same directory as the extracted MNIST dataset files.
4. Open a terminal or command prompt and navigate to the directory containing the `mnist_digit_recognizer.py` script.
5. Run the following command to train the neural network:

```
python mnist_digit_recognizer.py
```

6. The script will train the neural network for 10 epochs.
7. After the training is complete, the script will evaluate the accuracy of the model on the testing set.
8. The script will also make a prediction on a new handwritten digit image.

**Contributions**

Please feel free to contribute to this project by submitting pull requests with improvements or bug fixes.

**License**

This project is licensed under the MIT License.