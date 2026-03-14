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


from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
import torch.nn as nn

# BertForSequenceClassification already has classifier layer inside
# DO NOT add separate fc layer — that's what was causing the missing key error
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_classes)
        self.dropout = nn.Dropout(0.1)
        # NO self.fc here — BertForSequenceClassification handles classification internally

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


# Load sarcasm detector from HuggingFace
sarcasm_detector = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-irony",
    tokenizer="cardiffnlp/twitter-roberta-base-irony"
)

# Words that strongly indicate real sarcasm context
SARCASM_CONTEXT_WORDS = [
    'crash', 'breaks', 'broke', 'fail', 'failed', 'error', 'bug', 'worst',
    'terrible', 'horrible', 'awful', 'disaster', 'hate', 'useless', 'again',
    'seriously', 'really', 'thanks a lot', 'just what', 'exactly what',
    'wonderful', 'fantastic', 'perfect', 'brilliant', 'genius', 'lovely',
    'monday', 'traffic', 'meeting', 'boss', 'work', 'stuck', 'delay', 'late'
]

def is_sarcastic(text):
    words = text.lower().split()

    # Rule 1: Too short and no sarcasm context words — skip
    if len(words) <= 5:
        has_context = any(word in text.lower() for word in SARCASM_CONTEXT_WORDS)
        if not has_context:
            return False

    # Rule 2: Run the model and require high confidence
    result = sarcasm_detector(text[:512])[0]
    return result['label'] == 'irony' and result['score'] > 0.92


# Function to predict sentiment
def predict_sentiment(text, model, tokenizer, device='cpu', max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    label = preds.item()

    if label == 1:
        sentiment = "Positive Tweet"
    elif label == 0:
        sentiment = "Neutral Tweet"
    elif label == 2:
        sentiment = "Negative Tweet"
    else:
        sentiment = "Unknown Label"

    # Append sarcasm warning if detected
    if is_sarcastic(text):
        sentiment = sentiment + " ⚠️ (Sarcasm Detected — may be inaccurate)"

    return sentiment


# Load model and tokenizer
def load_model_and_tokenizer(model_path):
    device = torch.device('cpu')
    model = BERTClassifier('bert-base-uncased', num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer


# --- INITIALIZATION ---
output_path = 'twitter_sentiment_model.pth'
model, tokenizer = load_model_and_tokenizer(output_path)

def predict(text):
    return predict_sentiment(text, model, tokenizer)


# Simple test
if __name__ == "__main__":
    print(f"Normal:  {predict('I love football')}")
    print(f"Normal:  {predict('Modi is a very good PM.')}")
    print(f"Sarcasm: {predict('Oh great, another update that breaks everything. Just what I needed!')}")
    print(f"Sarcasm: {predict('I love it when my computer crashes right before I save my work')}")