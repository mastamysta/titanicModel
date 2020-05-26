<h1>titanicModel.py</h1>
<h2>Background</h2>
<p>
  Along with MNIST, the RMS Titanic survivor moel problem is one of the de facto 'Hellow World' programming tasks for machine learning. The objective of the task is to preprocess the records of the first 891 passengers including data such as cabin number and fare to produce a model with which to predict the survival of the remaining 418 passengers.
  The train and test datasets were taken from Kaggle: https://www.kaggle.com/c/titanic/
</p>
<br>
<h1>Code</h2>
<p>
  The code I wrote to clean and prepare the train and test data is somewhat messy. A separate for loop is used to iterate over the whole dataset each time a feature is processed where it would be easy to impliment a single for loop to iterate over the data and process all features in the same iteration. I opted not to do this to improve readability as this is not a performance critical task.
</p>
<br>
<h2>Reflections</h2>
<p>
The model was able to correctly predict which passengers survived at a rate of 68%. Obviously this rate is good but given this is a binary classification (and that a 61% accuracy could be acheived by classifying ALL members of the test set as deceased). To improve the accuracy, more fine tuning of the model itself is required. Perhaps a different network type or a tweak to the hyperparameters?
</p>
