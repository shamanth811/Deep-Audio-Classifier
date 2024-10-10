# Deep-Audio-Classifier
Languages Used: Python

Developed and implemented a deep audio classifier using CNNs and LSTMs to accurately categorize diverse audio signals, achieving high accuracy and robustness. Utilized Python and TensorFlow for model development and training, incorporating data augmentation techniques to enhance performance.

The code need the data file to be downloded from kaggel : link to the data : https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing

Key Components: 

a. Audio Preprocessing: Convert raw audio waveforms into spectrograms for processing by convolutional neural networks (CNNs).


b. Deep Learning Model: Train a CNN or recurrent neural network (RNN) on preprocessed audio data to learn features and classify audio into categories like speech, music or environmental sounds. c. Sliding Window Classification: Divide longer audio clips into shorter segments, apply the trained model to each segment, and aggregate the individual classifications to determine overall density of target audio events. d. Model Training and Optimization: Train the model on a diverse dataset using techniques like data augmentation and regularization to improve generalization.

By leveraging the power of deep learning, the deep audio classifier can significantly enhance the efficiency and accuracy of various audio processing tasks, making it a valuable tool in a wide range of industries.
