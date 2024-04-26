# Sentiment Analysis for Vietnamese Customer Feedback
This project is my assignment of Machine Learning course in my university, which requires students to learn and present an optional topic in AI/Machine Learning. 

The model is based on PhoBERT, a RoBERTa-based large language model for Vietnamese developed by VinAI, and is adapted for Customer Feedback Sentiment Analysis using Low-Rank Adaptation.

## Preprocessing dataset
To train the model, I have used a dataset downloaded from Kaggle in the following link: https://www.kaggle.com/datasets/linhlpv/vietnamese-sentiment-analyst

To remove the imbalanced data, I have deleted multiple positive-labeled rows as their quantity is overwhelmingly large compared to the others, reducing the dataset from 30k to 18k rows.

## Testing
The model was uploaded to HuggingFace in the following link, which you can download for any purpose: https://huggingface.co/datptm2003/lora-vietnamese-feedback-analysis



## References
**PhoBERT:** https://aclanthology.org/2020.findings-emnlp.92/

**LoRA:** https://arxiv.org/abs/2106.09685
