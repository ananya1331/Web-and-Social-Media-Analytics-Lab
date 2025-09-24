import spacy
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

texts = [
    "New Delhi is the capital of India.",
    "ISRO is planning to launch Gaganyaan in 2027",
    "F.R.I.E.N.D.S. is Ruchitankshi's favourite sitcom.",
    "My interests span Data Science to Cybersecurity.",
    "I would kill for a good burger right now, maybe even a momo!"
]

# Process each text separately
docs = [nlp(text) for text in texts]

# Initialize counters
pos_counts = {"NOUN": 0, "VERB": 0}
entity_counts = {"ORG": 0, "PERSON": 0, "GPE": 0}

# Count POS and entities
for doc in docs:
    for token in doc:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
    for ent in doc.ents:
        if ent.label_ in entity_counts:
            entity_counts[ent.label_] += 1

# Plot POS counts
plt.bar(pos_counts.keys(), pos_counts.values())
plt.title("POS Count")
plt.show()

# Optionally, print entity counts
print("Entity Counts:", entity_counts)

# Plot NER counts
plt.bar(entity_counts.keys(),entity_counts.values())
plt.title("NER Count")
plt.show()

#NER Counts
print("NER Counts:", entity_counts)