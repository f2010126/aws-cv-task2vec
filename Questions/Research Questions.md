## Extracting MetaFeatures from Training Data using Task2Vec

The workflow involves extracting the input features of data from intermediate layers of the model  and training the classifier on these features.
Then, the calculated gradients of this training are used to estimate the FIM and feature embeddings. 

***

### Which Feature Layers have the best information for the classifier?

The features obtained from lower layers of the model  are more generalised while higher layers focus on the details of the target task. Deep Models have a lot of layers and it's important to select to optimal range of layers to use to generate embeddings specifc to the dataset. 

#### Experiment 1

Vary the layer from where input features are cached for training the classifier. Compare the embeddings obtained for 2 similar, 1 Dissimilar dataset.  
(I expect the embedding generated using the earlier layers to yield a uniform distance matrix compared to when using the later layers.)
 
***

### **Is having a good classifier needed to generate meaningful embeddings?**

Good classifier here refers to measuring the accuracy of a model after the classifier has been fitted to the input features from intermediate layers. A good classifier will have the best possible performance on the test set. 

Meaningful Embeddings refers to information that will allow the model to learn similarities or dissimmilarities (represented by a distance metric) between datasets. 
Expectation would be a clearly defined distance matrix where the distance between datasets is in line with conventional understanding. (eg expecting a higher distance between datasets of different languages)


### Workflow
1. Introduce testing and validation data into the Task2Vec workflow
2. Measure the accuracy of the model after the classifier is fitted. Embeddings are extracted after training the classifier. Store the embeddings.

#### Experiment 1
Compare distance between different embeddings generated by the same probe network on the same dataset but with different Hyperparameter configs?
**How sensitive is the embedding generation process to HP? (I assume the distance matrix should be uniform, ie is robust to HP)**


#### Experiment 2
Obtain a classifier (model pipeline) that performs best accross the given datasets. Then introduce 2 dissimilar datasets and 3 similar datasets. 
**Are the embeddings obtained sufficiently different? (Clusterings should group the datasets accordingly)**

#### Experiment 3
Extent of distinction between tasks.
**If the classifier is sufficiently trained, can the embeddings distinguish finer points of Text classification?** 
Sufficiently trained, ie verge of overfitting to test data.
Eg: Topic classification vs Sentiment Analysis 

### Failure Case

***
Comment on: Can the accuracy of the classifier be seen as a proxy for the quality of the embedding?

