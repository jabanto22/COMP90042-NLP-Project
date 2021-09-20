# Rumour Detection and Analysis on Twitter
COMP90042 Natural Language Processing 2021 Semester 1 - Project

### Task1
Provided a dataset of source tweets and their replies, where each source tweet is labeled as a rumour or non-rumour, the task is to build a binary classifier using this dataset. For each tweet (source tweet or reply tweet), the dataset provides a range of information, including the text of the tweet, information of the user who made the
tweet, unique ID of the tweet, etc.

For this task, a sentence-pair binary classifier built on pre-trained BERT model was implemented using PyTorch.

### Task2
Using the trained rumour classifier from **Task1**, rumours from a COVID-19 set of tweets were detected to understand the popular hashtags and difference in setiments of rumour vs. non-rumour tweets.
