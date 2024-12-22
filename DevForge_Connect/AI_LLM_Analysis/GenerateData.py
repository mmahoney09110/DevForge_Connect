import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import re
from collections import Counter

# Load environment variables from the .env file
load_dotenv()

# Configure API key
genai.configure(api_key=os.getenv("GEMENI_API_KEY"))

# Initialize the Generative Model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=(
        "You are a data generation assistant. Generate JSON data in the following format: "
        '{"text": "A unique task.", "cats": {"CATEGORY": 0 or 1}}. '
        'Example: { "text": "Design and implement a new operating system for a self-driving car.", "cats": { "STORAGE": 1, "AI": 1, "DATA_ANALYSIS": 1, "COMMUNICATION": 1, "WEB_DEVELOPMENT": 0, "GAME_DEVELOPMENT": 0, "MOBILE_DEVELOPMENT": 0, "IOT": 1, "OPERATING_SYSTEMS": 1, "NETWORK": 1, "REALTIME": 1 } }'
        "Only generate data where WEB_DEVELOPMENT category is 1"
    )
)
chat = model.start_chat()

# Function to generate new responses from the model
def generate_response(prompt):
    print("Generating data...")
    try:
        response = chat.send_message(prompt)
        raw_response = response.text.strip()
        
        # Use regex to extract JSON content from the response
        match = re.search(r'\[\s*{.*}\s*\]', raw_response, re.DOTALL)
        if match:
            json_content = match.group(0)  # Extracted JSON content
            return json.loads(json_content)  # Parse the JSON
        else:
            print("Error: No valid JSON array found in the response.")
            return []

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

# Function to count label distribution
def count_labels(data):
    class_counts = Counter()
    for item in data["training_data"]:
        for label, value in item["cats"].items():
            if value:
                class_counts[label] += 1
    return class_counts

# Function to load training data
def load_training_data(file_path):
    if not os.path.exists(file_path):
        print("No existing training data found. Initializing new file...")
        return {"training_data": []}
    with open(file_path, "r") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            print("Error: Invalid JSON in training_data.json. Starting fresh.")
            return {"training_data": []}

# Function to save training data back to the file
def save_training_data(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
        print("Training data successfully saved.")

# Function to generate a balanced prompt
def create_balanced_prompt(label_counts):
    sorted_counts = sorted(label_counts.items(), key=lambda x: x[1])  # Sort by count
    overrepresented = [label for label, count in label_counts.items() if count > 500]  # Arbitrary threshold for overrepresentation
    underrepresented = [f"{label}: {count}" for label, count in sorted_counts[:5]]  # Focus on 5 lowest categories

    return (
        f"Generate 10 new examples in JSON format, each focusing on balancing the dataset. "
        f"Here are the underrepresented categories with their current counts: {', '.join(underrepresented)}. "
        f"Do not generate data for overrepresented categories: {', '.join(overrepresented)}."
    )

# Function to filter generated data
def filter_generated_data(new_data, label_counts, max_threshold=500):
    """Filter out generated data for overrepresented categories."""
    filtered_data = []
    for item in new_data:
        # Check if the generated example has valid categories and isn't overrepresented
        if all(label_counts[label] <= max_threshold or item["cats"].get(label, 0) == 0 for label in item["cats"]):
            filtered_data.append(item)
    return filtered_data

# Main Logic
def main():
    training_file = "training_data.json"

    # Step 1: Load existing training data
    training_data = load_training_data(training_file)

    # Step 2: Data generation loop
    for epoch in range(30):  # Number of iterations to generate new data
        label_counts = count_labels(training_data)
        print(f"Current Label Distribution: {label_counts}")

        # Generate a prompt with an explicit focus on underrepresented categories
        prompt = create_balanced_prompt(label_counts)

        # Step 3: Generate new data
        new_data = generate_response(prompt)

        # Step 4: Filter out overrepresented categories
        new_data = filter_generated_data(new_data, label_counts, max_threshold=500)

        if not new_data:
            print("No valid data generated after filtering. Skipping iteration.")
            continue

        print("Adding new data...")
        for item in new_data:
            # Ensure the generated data has valid format
            if "text" in item and "cats" in item:
                training_data["training_data"].append(item)
            else:
                print(f"Skipping invalid entry: {item}")

        # Step 5: Save updated training data after each iteration
        save_training_data(training_data, training_file)

    print("Data generation complete.")

if __name__ == "__main__":
    main()
