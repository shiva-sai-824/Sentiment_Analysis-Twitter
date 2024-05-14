# from transformers import BertTokenizer, BertModel
# import torch
# import torch.nn as nn


# class BERTClassifier(nn.Module):
#     def __init__(self, bert_model_name, num_classes):
#         super(BERTClassifier, self).__init__()
#         self.bert = BertModel.from_pretrained(bert_model_name)
#         self.dropout = nn.Dropout(0.1)
#         self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.pooler_output
#         x = self.dropout(pooled_output)
#         logits = self.fc(x)
#         return logits


# def predict_sentiment(text, model, tokenizer, device='cpu', max_length=128):
#     model.eval()
#     encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)

#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         _, preds = torch.max(outputs, dim=1)  # Assuming logits is the output of your model
#     label = preds.item()

#     if label == 2:
#         return "Positive Tweet"
#     elif label == 1:
#         return "Neutral Tweet"
#     elif label == 0:
#         return "Negative Tweet"
#     else:
#         return "Unknown Label"



# def predict(text):
#     return predict_sentiment(text, model, tokenizer)

# # Define model and load pre-trained weights
# model = BERTClassifier('bert-base-uncased', num_classes=3)  # Adjust num_classes according to your task
# model.load_state_dict(torch.load('https://drive.google.com/file/d/1pSIvYKih7JU9G746oxwp1w6IxfsuK3jk/view?usp=sharing', map_location=torch.device('cpu')))


# # Tokenizer initialization
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



# print(predict("modi is very good pm"))


from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import gdown

# Define BERT classifier model
# Define BERT classifier model
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

# Function to predict sentiment
def predict_sentiment(text, model, tokenizer, device='cpu', max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)  # Assuming logits is the output of your model
    label = preds.item()

    if label == 2:
        return "Positive Tweet"
    elif label == 1:
        return "Neutral Tweet"
    elif label == 0:
        return "Negative Tweet"
    else:
        return "Unknown Label"

# Function to predict sentiment for a given text
def predict(text):
    return predict_sentiment(text, model, tokenizer)

# Download model file from Google Drive
def download_model_from_drive(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

# Load model and tokenizer
def load_model_and_tokenizer(model_path):
    model = BERTClassifier('bert-base-uncased', num_classes=3)  # Adjust num_classes according to your task
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

# Download model from Google Drive
file_id = '14mjiuekEqZIJxbea6IudiLun9US5iIHz'  # Replace with your file ID
output_path = 'bert_classifier_three_labeled11.pth'
download_model_from_drive(file_id, output_path)

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(output_path)

# Example usage
print(predict("Modi is a very good PM."))  # Change the text as needed
