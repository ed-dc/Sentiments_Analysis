import pandas as pd
import matplotlib .pyplot as plt
import torch 
from nltk.corpus import stopwords
import nltk
import pickle

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from transformers import  BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW


from data_processing import clean_text
from keras.preprocessing.sequence import pad_sequences

#Config

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nltk.download('stopwords')

BATCH_SIZE = 32
EPOCHS = 5

#Load the data and visualisation 

df = pd.read_csv("data/archive/Combined_Data.csv")
df["statement"] = df["statement"].str.lower()
df = df.drop(columns=["Unnamed: 0"], axis=1)

df["statement"] = df["statement"].apply(clean_text)

# print(df.head())
list_count = list(df["status"].value_counts())
plt.bar(df["status"].value_counts().index, list_count)
plt.xlabel("Different Pathology")
plt.ylabel("Number of samples")
plt.title('Mental Health Status Counts')
# plt.show()

# print(x.head())

possibilities_status = df["status"].unique()
status_dict ={}

for idx, status in enumerate(possibilities_status):
    status_dict[status] = idx

df["status"] = df["status"].map(status_dict)


#separate to have train and test

x = df["statement"].values
y = df["status"].values

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case=True
)

def tokenize_and_stopwords(text):
    text = [word for word in text.split() if (word.lower() not in stopwords.words("english"))]
    text = ''.join(text)
    text = tokenizer.encode(text)
    return text


# Save tokenized data
# print("Tokenizing and saving data...")
# x = df["statement"].apply(tokenize_and_stopwords).values
# y = df["status"].values

# print(x[0], y[0])


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# with open('x_train_tokenize.pkl', 'wb') as f:
#     pickle.dump(x_train, f)

# with open('x_test_tokenize.pkl', 'wb') as f:
#     pickle.dump(x_test, f)

# with open('y_train.pkl', 'wb') as f:
#     pickle.dump(y_train, f)

# with open('y_test.pkl', 'wb') as f:
#     pickle.dump(y_test, f)

#Load tokenized data (example usage)

print("Loading tokenized data...")
with open('x_train_tokenize.pkl', 'rb') as f:
    x_train = pickle.load(f)

with open('x_test_tokenize.pkl', 'rb') as f:
    x_test = pickle.load(f)

with open('y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)

with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)


# Convert to tensors
print("Converting to tensors")

x_train_padded = pad_sequences(x_train, padding='post')
x_test_padded = pad_sequences(x_test, padding='post')

x_train_tensor = torch.tensor(x_train_padded)
y_train_tensor = torch.tensor(y_train)
x_test_tensor = torch.tensor(x_test_padded)
y_test_tensor = torch.tensor(y_test)

print("Creating DataLoader")

train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


#Model definition

print("Loading BERT model...")

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(possibilities_status)
)
model.to(device)

print("Model loaded successfully. and into device:", device)

optimizer = AdamW(model.parameters(), lr=2e-5)

#Training the model

def train_model(model, train_dataloader, optimizer, epochs=3):

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

#validation

def evaluate_model(model, test_dataloader):

    model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()

    print(f"Validation Loss: {total_loss / len(test_dataloader):.4f}")
    print(f"Validation Accuracy: {correct_predictions / len(test_dataloader.dataset):.4f}")

train_model(model, train_dataloader, optimizer, epochs=EPOCHS)
evaluate_model(model, test_dataloader)
