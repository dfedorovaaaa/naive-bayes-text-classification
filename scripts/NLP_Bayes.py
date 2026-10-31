import math
import re
from collections import Counter

from sklearn.datasets import fetch_20newsgroups

categories = [
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x"]

data = fetch_20newsgroups(subset="all", categories=categories, shuffle=False)

class_names = data.target_names
number_of_classes = len(class_names)

print("Категории:")

for i in range(number_of_classes):
    print(i, "->", class_names[i])
print()


documents_by_class = {}

for class_id in range(number_of_classes):
    documents_by_class[class_id] = []

for text, class_id in zip(data.data, data.target):
    documents_by_class[class_id].append(text)


train_documents = []
test_documents = []

for class_id in range(number_of_classes):

    documents = documents_by_class[class_id]

    split_index = int(len(documents) * 0.8)

    train_part = documents[:split_index]
    test_part = documents[split_index:]

    print(class_names[class_id], ": всего =", len(documents), "train =", len(train_part), "test =", len(test_part))

    for text in train_part:
        train_documents.append((text, class_id))

    for text in test_part:
        test_documents.append((text, class_id))


print("\nВсего train-документов:", len(train_documents))
print("Всего test-документов:", len(test_documents), "\n")

def preprocess_text(text):
    text = text.lower()
    words = re.findall(r"[a-z]+", text)
    return words

processed_train = []
for text, class_id in train_documents:
    words = preprocess_text(text)
    processed_train.append((words, class_id))

processed_test = []
for text, class_id in test_documents:
    words = preprocess_text(text)
    processed_test.append((words, class_id))


vocabulary = set()

for words, class_id in processed_train:
    for word in words:
        vocabulary.add(word)

V = len(vocabulary)

print("Размер словаря:", V, "\n")


def frequency_vector(words):
    frequencies = Counter()
    for word in words:
        if word in vocabulary:
            frequencies[word] += 1
    return frequencies

def binary_vector(words):
    present_words = set()
    for word in words:
        if word in vocabulary:
            present_words.add(word)
    return present_words

example_words, example_class = processed_train[0]

example_frequency = frequency_vector(example_words)
example_binary = binary_vector(example_words)

print("Пример документа")
print("Класс:", class_names[example_class])
print("Первые слова:")
print(example_words[:20])
print("Частотное представление:")
print(list(example_frequency.items())[:20])
print("Бинарное представление:")
print(list(example_binary)[:20])
print()

class_document_counts = Counter()
for words, class_id in processed_train:
    class_document_counts[class_id] += 1

total_train_documents = len(processed_train)

class_priors = {}
for class_id in range(number_of_classes):
    class_priors[class_id] = (class_document_counts[class_id] / total_train_documents)

print("Вероятности классов:")
for class_id in range(number_of_classes):
    print(class_names[class_id], "=", round(class_priors[class_id], 4))
print()

multinomial_word_counts = {}
multinomial_total_words = {}

for class_id in range(number_of_classes):
    multinomial_word_counts[class_id] = Counter()
    multinomial_total_words[class_id] = 0

for words, class_id in processed_train:
    frequencies = frequency_vector(words)
    for word, count in frequencies.items():
        multinomial_word_counts[class_id][word] += count
        multinomial_total_words[class_id] += count

multinomial_log_probabilities = {}
for class_id in range(number_of_classes):
    multinomial_log_probabilities[class_id] = {}
    total_words_in_class = multinomial_total_words[class_id]
    for word in vocabulary:
        word_count = multinomial_word_counts[class_id].get(word, 0)
        probability = ((word_count + 1) / (total_words_in_class + V))
        multinomial_log_probabilities[class_id][word] = (math.log(probability))

print("ТОП-10 САМЫХ ЧАСТЫХ СЛОВ ПО КЛАССАМ (Multinomial)")
for class_id in range(number_of_classes):
    print(f"\nКласс: {class_names[class_id]}")
    top_words = multinomial_word_counts[class_id].most_common(20)

    for word, count in top_words:
        print(f"  {word:15} -> {count} раз")

def predict_multinomial(words):
    frequencies = frequency_vector(words)
    best_class = None
    best_score = -float("inf")

    for class_id in range(number_of_classes):
        score = math.log(class_priors[class_id])

        for word, count in frequencies.items():
            score += (count * multinomial_log_probabilities[class_id][word])

        if score > best_score:
            best_score = score
            best_class = class_id

    return best_class


true_labels = []
multinomial_predictions = []

for words, real_class in processed_test:
    predicted_class = predict_multinomial(words)
    true_labels.append(real_class)
    multinomial_predictions.append(predicted_class)


bernoulli_document_counts = {}
for class_id in range(number_of_classes):
    bernoulli_document_counts[class_id] = Counter()

for words, class_id in processed_train:
    present_words = binary_vector(words)
    for word in present_words:
        bernoulli_document_counts[class_id][word] += 1


bernoulli_base_scores = {}
bernoulli_differences = {}

for class_id in range(number_of_classes):
    bernoulli_differences[class_id] = {}
    number_of_documents = class_document_counts[class_id]
    base_score = math.log(class_priors[class_id])

    for word in vocabulary:
        documents_with_word = (bernoulli_document_counts[class_id].get(word, 0))
        probability = ((documents_with_word + 1) / (number_of_documents + 2))

        log_present = math.log(probability)
        log_absent = math.log(1 - probability)

        base_score += log_absent
        bernoulli_differences[class_id][word] = (log_present - log_absent)

    bernoulli_base_scores[class_id] = base_score

def predict_bernoulli(words):
    present_words = binary_vector(words)
    best_class = None
    best_score = -float("inf")

    for class_id in range(number_of_classes):
        score = bernoulli_base_scores[class_id]

        for word in present_words:
            score += bernoulli_differences[class_id][word]

        if score > best_score:
            best_score = score
            best_class = class_id

    return best_class

bernoulli_predictions = []
for words, real_class in processed_test:
    predicted_class = predict_bernoulli(words)
    bernoulli_predictions.append(predicted_class)


def calculate_f1(true_labels, predicted_labels):
    f1_scores = {}

    for class_id in range(number_of_classes):
        TP = 0
        FP = 0
        FN = 0

        for real, predicted in zip(true_labels, predicted_labels):

            if real == class_id and predicted == class_id:
                TP += 1
            elif real != class_id and predicted == class_id:
                FP += 1
            elif real == class_id and predicted != class_id:
                FN += 1

        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)

        if TP + FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall / (precision + recall))

        f1_scores[class_id] = {"precision": precision, "recall": recall, "f1": f1}

    macro_f1 = 0

    for class_id in range(number_of_classes):
        macro_f1 += f1_scores[class_id]["f1"]

    macro_f1 /= number_of_classes

    return f1_scores, macro_f1


multinomial_scores, multinomial_macro_f1 = (calculate_f1(true_labels, multinomial_predictions))
bernoulli_scores, bernoulli_macro_f1 = (calculate_f1(true_labels, bernoulli_predictions))



print("\n\nMULTINOMIAL NAIVE BAYES\n")

for class_id in range(number_of_classes):
    scores = multinomial_scores[class_id]
    print(class_names[class_id])
    print("Precision:", round(scores["precision"], 4))
    print("Recall:", round(scores["recall"], 4))
    print("F1:", round(scores["f1"], 4))
    print()

print("\nMacro F1 Multinomial:", round(multinomial_macro_f1, 4))

print("\n\nBERNOULLI NAIVE BAYES\n")

for class_id in range(number_of_classes):
    scores = bernoulli_scores[class_id]
    print(class_names[class_id])
    print("Precision:", round(scores["precision"], 4))
    print("Recall:", round(scores["recall"], 4))
    print("F1:", round(scores["f1"], 4))
    print()

print("\nMacro F1 Bernoulli:", round(bernoulli_macro_f1, 4))

print("\n\nСРАВНЕНИЕ\n")
print("Multinomial:", round(multinomial_macro_f1, 4))
print("Bernoulli:", round(bernoulli_macro_f1, 4))

if multinomial_macro_f1 > bernoulli_macro_f1:
    print("Лучший результат: Multinomial Naive Bayes")
elif bernoulli_macro_f1 > multinomial_macro_f1:
    print("Лучший результат: Bernoulli Naive Bayes")
else:
    print("Модели показали одинаковый результат")
