# Anxiety_Prediction
## Overview
   Feelings of worry, nervousness, or unease about uncertain outcomes are usually associated with an emotion called anxiety. Anxiety is a feeling many people experience throughout their lives in varying degrees of intensity. However, there are individuals with anxiety so severe, it impedes their day-to-day functioning. Those with general anxiety disorder (GAD) experience this worry and unease chronically, and they are unable to control the behaviors that come along with it (Stein et al., 2015).
   
The DSM-IV highlights six symptoms of tension/negative affect of which to associate with symptoms of anxiety and to consider when making a diagnosis for GAD: restlessness or feeling on edge, being easily fatigued, difficulty concentrating or the mind going blank, irritability, muscle tension, and sleep disturbance (American Psychological Association, 1994). Three of the six symptoms (only one for children) must occur along with excessive anxiety being present for more days than not for at least 6 months to qualify. 

This iteration of the DSM also indicated these symptoms and worry connected to them must be perceived by the individual to be difficult to control. According to Brown and colleagues (2008), this revision from the previous edition of the DSM was made due to evidence showcasing the difference between the anxiety of the general population and those with GAD; while they both had worries about similar content, the controllability of the worry was reported to be vastly diminished in individuals with GAD (Brown et al., 2008). More criteria explain the anxiety, to qualify for a diagnosis, should cause distress and/or impairment in important functioning and should not be the result of substances or their side effects. 

In the DSM-V, the criteria have not made significant changes, but there is a short addition: “the disturbance should not be better explained by another mental disorder” (American Psychiatric Association, 2013).

The prevalence of GAD in the general population ranges from 1.9% to 5.4% (Brown et al., 2008). The prevalence in the United States was 3.1% in 2014 and 5.7% over the course of a patient’s lifetime, according to epidemiological surveys (Stein et al., 2015). While the age of onset is highly variable, there is a higher rate of females having GAD compared to males at a rate of 2:1 (Stein et al., 2015). Risk factors of GAD include low socioeconomic status, exposure to childhood adversity, and being female; twin studies have also shown anxiety has a moderate chance (about 15-20%) of being inherited through genetics (Stein et al., 2015).

General anxiety disorder also commonly co-occurs with major depressive disorder, with many of their symptoms overlapping and thus making it difficult to distinguish the two diagnoses (Stein et al., 2015). Although the inability to experience pleasure does not overlap with GAD, individuals with it feel hopelessness like patients with major depressive disorder. According to the NCS, 65% of patients with GAD also had at least one other disorder at the time of their assessment (Brown et al., 2008).

Additionally, patients with GAD have a higher risk for other mental health disorders and physical symptoms such as chronic pain and asthma (Stein et al., 2015). About 35% of those with GAD turn to alcohol and medications to reduce their symptoms of anxiety, which can increase the risk for substance and drug related problems (Stein et al., 2015). 

## Import Libraries
![image](https://user-images.githubusercontent.com/90658957/186297635-74d1b2c1-3c28-42ce-9ba2-40daf72de611.png)

## EDA
## Distribution of text length in user messages
![image](https://user-images.githubusercontent.com/90658957/186297957-dc1658e2-1cf1-4cee-8ded-1cbe95b70a93.png)

## wordcloud
![image](https://user-images.githubusercontent.com/90658957/186298042-7bb2d1d9-e68f-4335-8a74-6a777d4d2a24.png)

## Sentiment Analysis
![image](https://user-images.githubusercontent.com/90658957/186298182-0f51a48f-68c5-43e3-bcd9-d1593bed4214.png)

## Top words after stopwords removal
![image](https://user-images.githubusercontent.com/90658957/186299018-ceb3fb38-b85a-4bcb-9375-a71d421ef999.png)


## Models
## Feed Forward Neural network using keras:
 
Keras is an open-source python framework used for creating and analyzing deep learning models. It is part of the TensorFlow library and allows us to define and train neural network models. After loading the dataset, we split the data into input (X) and output (y) variables and then create a Sequential model and add layers to our network architecture. Fully connected layers are defined using the Dense class. we can specify the number of neurons or nodes in the layer as the first argument and the activation function using the activation argument. Also, we will use the rectified linear unit activation function referred to as 'relu' on the first two layers and the Sigmoid function in the output layer. By using a sigmoid on the output layer, we can easily transfer our network output to a probability of class 1, or, with a default threshold of 0.5, snap to a hard classification of either class. After adding the layers, we will compile the model because it has been specified. For training and producing predictions on our hardware, such as CPU, GPU, or even distributed, the backend automatically determines the appropriate method to represent the network. There are a few more characteristics that must be specified during compilation in order to train the network. Keeping this in mind, determining the optimal set of weights to translate our dataset's inputs to outputs while training a network.

We used cross entropy as the loss justification. This loss, known in Keras as "binary crossentropy,". We will use the effective stochastic gradient descent method "adam" to define the optimizer. This variant of gradient descent is well-liked since it automatically fine-tunes itself and produces effective solutions to a variety of issues. The classification accuracy described by the metrics argument will be collected and reported because it is a classification problem. Now we run the model on some data. By using the fit() method on the model, we will train or fit our model using the loaded data. The training process will run for a fixed number of epochs (iterations) through the dataset that will be specified using the epochs argument. We used the whole dataset to train our neural network, then the validation dataset to assess its performance. The evaluate() function will return loss, accuracy, f1, precision and recall for validation dataset.

## Convolutional Neural Network (CNN):

In Keras, we may simply add the necessary layer one at a time to build up layers. The Sequential object's add method is then called to add layers. The layers themselves are examples of classes like Dense, which denotes a layer that is fully linked and uses a certain amount of neurons with a certain activation function.
That is exactly what we did in this case: we added a first convolutional layer using Conv1D (). The rectified linear unit activation function, often known as relu, will then be used on the first layer. Next, we’ll add the max-pooling layer with MaxPooling1D() and so on. The last layer is a dense layer that sigmoid activation. After the model is created, we compile it using Adam optimizer, one of the most popular optimization algorithms and we used cross entropy as the loss justification. This loss, known in Keras as "binary crossentropy,". Lastly, we specify the metrics as accuracy which we want to analyze while the model is training. We use summary function to visualize what we have done above. By using the fit () method on the model, we will train or fit our model using the loaded data. By observing the training accuracy and loss we can say that the model did a good job or not. We used the whole dataset to train our neural network, then the validation dataset to assess its performance. The evaluate () function will return loss, accuracy, f1, precision and recall for validation dataset.

## Model Performance 
## Feed Forward Neural Network using keras
![image](https://user-images.githubusercontent.com/90658957/186298572-040f169b-731f-4061-a40c-531de94257c4.png)

## CNN
![image](https://user-images.githubusercontent.com/90658957/186298668-f2d812bf-5437-4996-8655-86c71732d230.png)

## Random Forest Regressor
![image](https://user-images.githubusercontent.com/90658957/186298747-e7363d04-94bf-4571-b01e-c6ed0fa775ff.png)

## Acknowledgements

   
