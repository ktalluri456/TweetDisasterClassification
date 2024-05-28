import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
import re

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

def clean_text_simple(text: str) -> str:
    """
    Cleans a given text by removing unwanted characters and converting it to lower case.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.

    This function removes:
        - URLs
        - User handles (@username)
        - Hashtags (#hashtag)
        - Non-alphanumeric characters (e.g. !, @, #, etc.)
        - Converts the text to lower case
    """
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user handles (@username)
    text = re.sub(r'\@\w+', '', text)
    # Remove hashtags (#hashtag)
    text = re.sub(r'#', '', text)
    # Remove non-alphanumeric characters (e.g. !, @, #, etc.)
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    # Convert the text to lower case
    text = text.lower()
    return text

train_data['clean_text'] = train_data['text'].apply(clean_text_simple)
test_data['clean_text'] = test_data['text'].apply(clean_text_simple)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TweetDataset(Dataset):
    """
    A custom dataset class for tweets. This class encapsulates the logic for
    encoding and batching tweets for training and evaluation.

    Args:
        texts (List[str]): A list of tweet texts.
        targets (List[int], optional): A list of corresponding labels.
            Defaults to None.
    """

    def __init__(self, texts, targets=None):
        """
        Initialize the TweetDataset object.

        Args:
            texts (List[str]): A list of tweet texts.
            targets (List[int], optional): A list of corresponding labels.
                Defaults to None.
        """
        self.texts = texts
        self.targets = targets
        # Encode the text data using the Hugging Face BERT tokenizer
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)

    def __getitem__(self, idx):
        """
        Get the encoded representation of a tweet at a given index.

        Args:
            idx (int): The index of the tweet to retrieve.

        Returns:
            dict: A dictionary containing the encoded tweet data. If the labels
                are available, the dictionary also contains the labels for the
                tweet.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # If labels are available, include them in the returned dictionary
        if self.targets is not None:
            item['labels'] = torch.tensor(self.targets[idx])
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.texts)

X_train, X_val, y_train, y_val = train_test_split(train_data['clean_text'], train_data['target'], test_size=0.2, random_state=42)
train_dataset = TweetDataset(X_train.tolist(), y_train.tolist())
val_dataset = TweetDataset(X_val.tolist(), y_val.tolist())
test_dataset = TweetDataset(test_data['clean_text'].tolist())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = len(np.unique(y_train)))

def train(model, train_loader, val_loader, epochs=3):
    """
    Train the model on the given dataset using the given hyperparameters.

    Args:
        model: The model to be trained.
        train_loader: A DataLoader object containing the training data.
        val_loader: A DataLoader object containing the validation data.
        epochs: The number of epochs to train the model. Defaults to 4.

    Returns:
        None
    """
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)  # AdamW optimizer with LR=0.001
    model.train()  # Set the model to training mode

    for epoch in range(epochs):
        total_loss = 0  # Initialize the total loss for this epoch
        for batch in train_loader:
            optimizer.zero_grad()  # Reset the gradients
            outputs = model(**batch)  # Forward pass
            loss = outputs.loss  # Get the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the parameters
            total_loss += loss.item()  # Accumulate the loss
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')  # Print the loss for this epoch

        model.eval()  # Set the model to evaluation mode
        val_preds, val_labels = [], []  # Initialize the lists to store the predictions and labels
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(**batch)  # Forward pass
                logits = outputs.logits  # Get the logits
                val_preds.extend(torch.argmax(logits, dim=-1).tolist())  # Get the predictions
                val_labels.extend(batch['labels'].tolist())  # Get the labels
        accuracy = accuracy_score(val_labels, val_preds)  # Calculate the accuracy
        f1 = f1_score(val_labels, val_preds)  # Calculate the F1 score
        print(f'Validation Accuracy: {accuracy}, F1-Score: {f1}')  # Print the validation metrics

train(model, train_loader, val_loader)

model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        outputs = model(**batch)
        logits = outputs.logits
        test_preds.extend(torch.argmax(logits, dim=-1).tolist())

submission = pd.DataFrame({
    'id': test_data['id'],
    'target': test_preds
})

submission.to_csv('submission.csv', index=False)

print(submission.head())
