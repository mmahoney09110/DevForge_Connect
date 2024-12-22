import spacy
from spacy.training import Example
import random
import json
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import spacy.util

# Load the config file
config_path = "config.cfg"
config = spacy.util.load_config(config_path)

# Initialize the NLP object with the configuration
nlp = spacy.blank("en")
nlp.from_config(config)  # This loads the configuration into the nlp object

# Add the text categorizer to the pipeline
if "textcat_multilabel" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat_multilabel", last=True)

# Add labels for the text categorizer
labels = ["STORAGE", "AI", "DATA_ANALYSIS", "COMMUNICATION", "WEB_DEVELOPMENT",
          "GAME_DEVELOPMENT", "MOBILE_DEVELOPMENT", "IOT", "NETWORK", "OPERATING_SYSTEMS", "REALTIME"]

for label in labels:
    textcat.add_label(label)

# Initialize the model after adding the textcat and labels
nlp.initialize()

# Load training data
with open('training_data.json', 'r') as file:
    TRAINING_DATA_FILE = json.load(file)

# Extract the training data
TRAINING_DATA = [(item["text"], {"cats": item["cats"]}) for item in TRAINING_DATA_FILE["training_data"]]

# Split the dataset into training and validation sets
TRAINING_DATA, VALIDATION_DATA = train_test_split(TRAINING_DATA, test_size=0.33, random_state=42)

# Check class distribution in training and validation sets
train_classes = [item[1]['cats'] for item in TRAINING_DATA]
valid_classes = [item[1]['cats'] for item in VALIDATION_DATA]

train_class_counts = Counter([label for cat in train_classes for label, value in cat.items() if value])
valid_class_counts = Counter([label for cat in valid_classes for label, value in cat.items() if value])

print(f"Training set class distribution: {train_class_counts}")
print(f"Validation set class distribution: {valid_class_counts}")

# Initialize the optimizer after the pipeline is set up
optimizer = nlp.begin_training()

# Function to evaluate the model
def evaluate_model(nlp, validation_data):
    preds, truths = [], []
    for text, annotations in validation_data:
        doc = nlp(text)
        preds.append({k: v >= 0.5 for k, v in doc.cats.items()})
        truths.append(annotations["cats"])

    # Transforming to binary matrices after the loop
    y_true = np.array([[truth[key] for key in labels] for truth in truths])
    y_pred = np.array([[int(pred[key]) for key in labels] for pred in preds])

    print(classification_report(y_true, y_pred, zero_division=0))

# Training loop
for epoch in range(36):
    random.shuffle(TRAINING_DATA)
    losses = {}
    for text, annotations in TRAINING_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], sgd=optimizer, losses=losses)
    print(f"Epoch {epoch} - Losses: {losses}")
    
    # Evaluate on validation data
    if epoch % 5 == 0:
        print(f"Validation Evaluation after Epoch {epoch}:")
        evaluate_model(nlp, VALIDATION_DATA)

# Save the trained model
nlp.to_disk("trained_models")
