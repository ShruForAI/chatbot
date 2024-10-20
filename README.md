# Generative Chatbot Project

This repository contains the code and resources for the final university project on building a generative chatbot using the Cornell Movie Dialog dataset. The project includes detailed exploratory data analysis (EDA), model training, and evaluation of a chatbot that generates conversational responses.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Architecture](#model-architecture)
- [Challenges Faced](#challenges-faced)
- [How to Run the Project](#how-to-run-the-project)
- [Model File](#model-file)
- [References](#references)

## Project Overview
The goal of this project was to develop a generative chatbot using natural language processing (NLP) techniques and deep learning. The chatbot was trained on the Cornell Movie Dialog dataset, which contains a rich collection of movie character conversations.

## Dataset
The Cornell Movie Dialog dataset was used for this project. This dataset includes:
- **movie_lines.txt**: Contains the dialogues between characters in various movies.
- **movie_conversations.txt**: Contains the conversations, which are pairs of dialogue lines between characters.

You can find the dataset on Kaggle [here](https://www.kaggle.com/datasets/Cornell-University/movie-dialog-corpus).

## Exploratory Data Analysis (EDA)
A thorough EDA was performed to understand the structure of the dataset and gain insights into the dialogues and conversations. The EDA includes:
- Distribution of dialogue lengths
- Top common words in dialogues
- Word clouds of the dialogue corpus
- Conversation length analysis
- Frequent question-response pairs

Visualizations such as bar charts, word clouds, and heatmaps were generated to support the findings.

## Model Architecture
The chatbot was built using a sequence-to-sequence (Seq2Seq) architecture. The model includes an encoder-decoder structure with LSTM layers. The encoder processes the input dialogue, and the decoder generates a response based on the encoded context.

Due to computational and memory limitations, a sample of 30,000 records was used for training the model.

## Challenges Faced
One of the main challenges was the size of the dataset and the resource constraints. Due to these limitations, it was not possible to fine-tune larger pre-trained models or process the entire dataset. Instead, a subset of the data was used to train the model.

## How to Run the Project
Run the Jupyter notebook files to explore the dataset, perform EDA, and train the model:

EDA.ipynb: Contains all the exploratory data analysis code.
chatbot_training.ipynb: Contains the code for training the chatbot model.

Install the required dependencies:
pip install -r requirements.txt

To test the chatbot locally, run the web interface:
python app.py

## Model File
The trained model (chatbot_model_2.h5) is 1.04GB, which exceeds the file upload limits for GitHub. As a result, the model is not included in this repository. If you wish to obtain the model, please contact me at akshruthi.official@gmail.com.

## References
Cornell Movie Dialog Dataset: [Kaggle Dataset](https://www.kaggle.com/datasets/rajathmc/cornell-moviedialog-corpus)

Sequence-to-Sequence Model: [Paper by Sutskever et al](https://arxiv.org/abs/1409.3215).
