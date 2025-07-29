import json

# From your training script, you know that 0 is "Cat" and 1 is "Dog"
class_indices = {'Cat': 0, 'Dog': 1}
# We want to map the output index back to the label name
labels = {v: k for k, v in class_indices.items()}
# labels will be {0: 'Cat', 1: 'Dog'}

with open('labels.json', 'w') as f:
    json.dump(labels, f)