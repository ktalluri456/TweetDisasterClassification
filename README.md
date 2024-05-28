# TweetDisasterClassification

## Introduction
This project is a machine learning project that classifies tweets as either a tweet that actually describes a natural disaster happening. The dataset used in this project is from [kaggle](https://www.kaggle.com/c/nlp-getting-started), and the dataset is from [kaggle.com](https://www.kaggle.com/c/nlp-getting-started/data).

## Model Choice
For this program, I used the [Hugging Face BERT model](https://huggingface.co/bert-base-uncased) with the [PyTorch](https://pytorch.org/) framework. The model used was trained with the [Hugging Face Transformers](https://huggingface.co/transformers/) library.

## Required Libraries

- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Hugging Face BERT model](https://huggingface.co/bert-base-uncased)

## How to Run
To run the program, simply run `python3 disaster_prediction.py` from the terminal.