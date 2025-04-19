import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from transformers_interpret import SequenceClassificationExplainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocess the dataset
def preprocess_dataset(filepath):
    # Read the dataset and handle empty lines
    df = pd.read_csv(filepath, skip_blank_lines=True)
    
    # Save the original DataFrame for future reference
    original_df = df.copy()
    
    # Drop rows where the 'Email Text' or 'Email Type' is missing
    df = df.dropna(subset=['Email Text', 'Email Type'])
    
    # Map Email Type: 0 for Safe Email, 1 for Phishing Email
    df['Email Type'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
    
    return df, original_df

# Define a custom Dataset class for your email dataset
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
# Load and preprocess the dataset
df, original_df = preprocess_dataset('Phishing_Email.csv')
df = df[['Email Text', 'Email Type']]

# Check if indices 2816 and 11526 exist before dropping
sample_text_1 = original_df['Email Text'].iloc[2816]
sample_text_2 = original_df['Email Text'].iloc[12039]
 
# Split the dataset into training and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['Email Text'], df['Email Type'], test_size=0.3, random_state=42)

# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

model.to(device)

# Create datasets and data loaders
train_dataset = EmailDataset(train_texts.to_numpy(), train_labels.to_numpy(), tokenizer, max_len=512)
test_dataset = EmailDataset(test_texts.to_numpy(), test_labels.to_numpy(), tokenizer, max_len=512)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

def train_model(train_loader, model, optimizer, epochs=3, device=None):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
        
# Define evaluation function
def evaluate_model(test_loader, model, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Get model predictions and calculate loss
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Get predicted labels
            preds = torch.argmax(outputs.logits, dim=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')

    # Generate classification report
    target_names = ['Safe Email', 'Phishing Email']
    report = classification_report(all_labels, all_preds, target_names=target_names)
    print('Classification Report:\n', report)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

    return avg_loss, accuracy

# LIME explanation function
def lime_explanation(model, text, class_names):
    explainer = LimeTextExplainer(class_names=class_names)
    
    def predict_proba(texts):
        encodings = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=128,  # Reduce token limit
            return_tensors="pt"
        )
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return probabilities.cpu().numpy()
    
    # Use num_features=5 to reduce the complexity
    explanation = explainer.explain_instance(text, predict_proba, num_features=5, top_labels=1)
    exp_class = explanation.available_labels()[0]
    exp = explanation.as_list(label=exp_class)
    explanation.show_in_notebook()
    
    return exp

# Transformer interpretability function
def transformer_interpret(model, tokenizer, text):
    # Tokenize and truncate the text to the model's max length
    encoding = tokenizer.encode_plus(
        text,
        truncation=True,  # Enable truncation
        max_length=512,   # Set the maximum length to 512 tokens
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Perform interpretability analysis with truncated text
    cls_explainer = SequenceClassificationExplainer(model, tokenizer)
    word_attributions = cls_explainer(text[:512])  # Truncate the text for interpretability
    cls_explainer.visualize()

    return word_attributions

# Explanation function that combines LIME and Transformer interpretability
def explain_prediction(model, tokenizer, text):
    class_names = ['Safe Email', 'Phishing Email']
    
    # LIME Explanation
    print("LIME Explanation:")
    lime_exp = lime_explanation(model, text, class_names)
    print(lime_exp)
    
    # Transformer Interpret
    print("Transformer Interpretability:")
    transformer_exp = transformer_interpret(model, tokenizer, text)
    print(transformer_exp)

#Train and evaluate the model
print("Training the model...")
train_model(train_loader, model, optimizer, epochs=4)

print("Evaluating the model...")
evaluate_model(test_loader, model, device)

# Explain the first email sample
print("\nExplaining the first email sample:\n")
explain_prediction(model, tokenizer, sample_text_1)

# Explain the second email sample
print("\nExplaining the second email sample:\n")
explain_prediction(model, tokenizer, sample_text_2)
