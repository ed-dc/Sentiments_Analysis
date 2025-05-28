### Sentiments_Analysis

## Description :

It's an Nvidia Course program, about NLP and in this case Sentiments Analysis based on a text.

## Setup

You can run when creating a new venv with : 

```
python3 -m venv venv
```

and then you need to install the requirements : 
```
pip install -r requirement.txt
```

## DataSet:

I used the Sentiment Analysis for Mental Health dataset from Kaggle(https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/code) wich has different status for different phrases : 
- Normal
- Depression
- Suicidal
- Anxiety
- Stress
- Bi-Polar
- Personality Disorder

And I used Pandas to process the data.


## Model :

We'll exclude stop words in our NLP Process. 

I choosed to use BertForSequenceClassification model and retrain it on our dataset.



## Problems encountered: 

I choosed to use BERTforSequenceClassification but the inputs when tokenized are too long => when removing stopwords it fitted

I had to pad_sequences the different x_train, x_test in order to always have the same length datas.


