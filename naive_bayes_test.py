from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from naive_bayes_PES1UG22CS168 import NaiveBayesClassifier
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def run_tests(test_cases):
    # Defining the training sentences and categories
    sentences = [
        "The new smartphone has a stunning display and battery life.",  # Technology
        "This pizza recipe is easy and delicious.",  # Food
        "The latest movie has been a box office hit.",  # Entertainment
        "AI is transforming the future of work.",  # Technology
        "This pasta dish is perfect for family dinners.",  # Food
        "The concert was an unforgettable experience.",  # Entertainment
        "5G networks are rolling out across the globe.",  # Technology
        "Chocolate cake is my favorite dessert.",  # Food
        "The streaming platform has released new shows.",  # Entertainment
        "Electric cars are the future of transportation.",  # Technology
        "The ice cream truck arrived on a hot day.",  # Food
        "Missing label",  # Missing label (noise)
        "Virtual reality is reshaping gaming.",  # Technology
        "Incorrect label"  # Incorrect label (noise)
    ]

    categories = [
        "technology", "food", "entertainment", "technology", "food", "entertainment", "technology",
        "food", "entertainment", "technology", "food", None, "technology", "wrong_label"
    ]

    # Preprocessing step
    sentences, categories = NaiveBayesClassifier.preprocess(sentences, categories)

    # Vectorizing the text data using CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(sentences)

    # Fitting the Naive Bayes model
    class_probs, word_probs = NaiveBayesClassifier.fit(X_train_vec.toarray(), categories)

    num_passed = 0

    for test_sentence, correct_category in test_cases:
        test_vector = vectorizer.transform([test_sentence]).toarray()
        prediction = NaiveBayesClassifier.predict(test_vector, class_probs, word_probs, np.unique(categories))[0]

        if prediction == correct_category:
            print(f"Test Passed: '{test_sentence}' - Predicted: {prediction} | Correct: {correct_category}")
            num_passed += 1
        else:
            print(f"Test Failed: '{test_sentence}' - Predicted: {prediction} | Correct: {correct_category}")

    return num_passed


if __name__ == "__main__":
    # Defining the test cases
    test_cases = [
        ("The smartphone has amazing battery life.", "technology"),
        ("I love this chocolate cake recipe.", "food"),
        ("The movie was a box office hit.", "entertainment"),
        ("Virtual reality is transforming gaming.", "technology"),
        ("This is the best pizza I've ever had.", "food"),
        ("The concert was mind-blowing.", "entertainment"),
        ("AI will shape the future of transportation.", "technology"),
        ("The ice cream was perfect on a summer day.", "food"),
        ("A blockbuster movie with a thrilling plot.", "entertainment"),
        ("The latest advancements in electric cars are impressive.", "technology")
    ]

    num_passed = run_tests(test_cases)
    print(f"\nNumber of Test Cases Passed: {num_passed} out of {len(test_cases)}")