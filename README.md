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

## Literature Overview
Machine learning algorithms have been present in the process of diagnosing and predicting future outcomes related to mental health.  The science field has used machine learning algorithms for a variety of areas because they can apply solutions without human input (Arif et al., 2020).They have also been used to detect the prognosis of various mental health disorders such as bipolar disorders and panic disorder (Pintelas, 2018). In the context of anxiety disorders, these machines have been used to predict and detect anxiety. More specifically, they can potentially be used to diagnose anxiety, predict future risk of anxiety, and predict responses to medical treatment (Arif et al., 2020). 

Different types of machine learning algorithms have been tested in the context of predicting anxiety disorders. Prior studies have made use of a Bayesian network, a probable graphic model that represents certain attributes amongst others, Artificial Neural Networks (also known as ANNs), adaptive processing units made for discovering new knowledge, Support Vector Machines (also known as SVM), supervised learning models meant for analyzing and classifying data based on training data it has been provided with, decision trees, they predict responses and branches based on specific features present in data, linear regression (also known as LR), explains the relationship between the outcome and another variable, and Neuro-Fuzzy Systems (also known as NFSs), combines neural networking and fuzzy logic to develop new fuzzy rules or functions of the inputs and outputs in the system (Pintelas et al., 2018). Individually, these algorithms and machines can do little. This is why the scientific community has turned to hybrid models, models that combine two or more existing methods to create a more efficient product (Pintelas et al., 2018). 

	Pintelas and colleagues (2018) gathered five studies that specifically looked at how different machine algorithms predicted generalized anxiety disorder and noted their main findings. Chatterjee and colleagues (2014) made use of logistic regression, Naive Bayes, and a Bayesian Network while feeding the machines input data based on inferred heart-rate measurements. Chen and colleagues (2015) used a Bayesian joint model paired with a linear mixed effects model and a generalized linear model to analyze input data from self-esteem data and anxiety diagnosis in regards to examining the development of self-esteem on adult onset anxiety disorder. Dabek and colleagues (2015) used ANN to analyze a dataset of patients. Katsis and colleagues (2011) used ANN, RF, NFS, and SVM to predict affective states of an individual based on five defined classes without any input data. Hilbert et al. (2017) used a SVM nestled within a leave-one-out-cross-validation framework to separate GAD diagnoses from healthy subjects and major depressive disorder with input data from questionnaires, cortisol release and white and grey matter volumes.
	
	Chaterjee and colleagues (2014) found the Bayesian Network model was the most accurate machine learning algorithm they had tested with an accuracy of 73.33%. Chen and colleagues (2015) found the joint-model to be more effective with a 75% accuracy rate. Dabek and colleagues (2015) found an overall 82.35% accuracy rate using ANN. Katsis and colleagues (2011) found the NFS to be the most accurate than the other models with a 84.3% being the highest accuracy level. Taking all the data into account, Hilbert and colleagues (2017) found an improved accuracy in detecting GAD from healthy individuals and differentiating it from major depressive disorder at 90.10% and 67.46% respectively. According these results, ANN was concluded to be the most accurate in predicting GAD (Pintelas et al., 2018).
	
	Further studies have given rise to more accurate machine learning algorithms in detecting anxiety. Sribala and colleagues (2015) were able to develop a neural network -based model capable of predicting GAD via the use of attributes listed in the DSM-IV standard questionnaire with an accuracy of 90.32% (Arif et al., 2020). With the sensitivity analysis taken into account, the accuracy of prediction increases to 96.43%. Another researcher, Hussain and colleagues (2016), used data from the Beck Depression Inventory (a questionnaire to measure the severity of depression) to make a GAD data set and apply a random forest tree. The best result from this tree, balanced data and 100 trees, resulted in highly accurate predictivity outcome of 99.3% (Arif et al., 2020).

## Acknowledgements
1. https://www.nejm.org/doi/full/10.1056/nejmcp1502514?casa_token=AaWL11SOxSwAAAAA:jF5Oz0b-b-QU5BEr0PUtrSIWN4p4VtcDOoyGoIcNkfh1cHLISUtCjdgeA1BrI8BAewvn64X5HoirCXQ 
2. https://www.researchgate.net/profile/Mark-Oakley-Browne/publication/230794754_Generalized_anxiety_disorder/links/09e415121f4e5ed7f1000000/Generalized-anxiety-disorder.pdf  
3. https://dl.acm.org/doi/abs/10.1145/3218585.3218587?casa_token=T-5L0Pbg7wAAAAAA:Zlc09PrHmhOL3wWxtZUHQaJa3U7km2T9ZOgqpQumwLRMbEy0wbbYBBZkzVFqn40OrcCMkfva2TdT 
Pintelas, E. G., Kotsilieris, T., Livieris, I. E., & Pintelas, P. (2018, June). A review of machine learning prediction methods for anxiety disorders. In Proceedings of the 8th International Conference on Software Development and Technologies for Enhancing Accessibility and Fighting Info-exclusion (pp. 8-15).
4. https://pdfs.semanticscholar.org/f63f/4a24f06ea76155607c483a664135af956b4d.pdf 
Arif, M., Basri, A., Melibari, G., Sindi, T., Alghamdi, N., Altalhi, N., & Arif, M. (2020). Classification of anxiety disorders using machine learning methods: a literature review. Insights Biomed Res, 4(1), 95-110.


   
