import numpy as np
import warnings
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

warnings.filterwarnings("ignore", category=RuntimeWarning)

class NaiveBayesClassifier:
    """
    A simple implementation of the Naive Bayes Classifier for text classification.
    """

    @staticmethod
    def classify(documents: np.ndarray, class_probabilities: dict, word_probabilities: dict, unique_classes: np.ndarray) -> list:
        """
        Classifies the input documents based on trained probabilities.
        
        Parameters:
        - documents: np.ndarray - The documents to classify (binary representation of word presence).
        - class_probabilities: dict - The prior probabilities for each class.
        - word_probabilities: dict - The conditional probabilities for words given the classes.
        - unique_classes: np.ndarray - List of unique classes.
        
        Returns:
        - list - Predicted classes for each document.
        """
        predictions = []

        for doc in documents:
            log_probs = defaultdict(float)  # Default to 0
            
            for cls in unique_classes:
                

                log_probs[cls] = np.log(class_probabilities[cls])
                
                
                present_words = np.where(doc)[0]


                log_probs[cls] += np.sum([np.log(word_probabilities[cls].get(word_idx, 1e-10)) for word_idx in present_words])




            predicted_cls = max(log_probs, key=log_probs.get)



            predictions.append(predicted_cls)

        return predictions

    @staticmethod
    def clean_data(sentences: list, labels: list) -> tuple:
        """
        Cleans the dataset by removing stop words and invalid labels.
        
        Parameters:
        - sentences: list - Input sentences to be cleaned.
        - labels: list - Corresponding labels for each sentence.
        
        Returns:
        - tuple - A tuple containing cleaned sentences and labels.
        """
        filtered_sentences = []

        filtered_labels = []

        for sentence, label in zip(sentences, labels):
            words = sentence.lower().split()
            cleaned_words = [word for word in words if word not in ENGLISH_STOP_WORDS]
            

            
            if cleaned_words and label:  
                filtered_sentences.append(' '.join(cleaned_words))
                filtered_labels.append(label)

        return filtered_sentences, filtered_labels

    @staticmethod
    def train_model(documents: np.ndarray, labels: np.ndarray) -> tuple:
        """
        Trains the Naive Bayes Classifier using the provided data.
        
        Parameters:
        - documents: np.ndarray - Training documents in binary format.
        - labels: np.ndarray - Corresponding labels for the documents.
        
        Returns:
        - tuple - Prior probabilities and conditional probabilities for the classes.
        """
        
       

        class_counts = Counter(labels)
        total_documents = len(labels)

        

        word_counts_per_class = {cls: Counter() for cls in class_counts}

       


        for index in range(total_documents):
            cls = labels[index]
            word_counts_per_class[cls].update(np.where(documents[index])[0])

       

        class_probabilities = {cls: count / total_documents for cls, count in class_counts.items()}

      

        total_words_in_class = {cls: sum(word_counts_per_class[cls].values()) + len(word_counts_per_class[cls]) for cls in word_counts_per_class}

        word_probabilities = {}

        for cls in word_counts_per_class:
            word_probabilities[cls] = {
                word_idx: (count + 1) / total_words_in_class[cls]  
                for word_idx, count in word_counts_per_class[cls].items()
            }

        return class_probabilities, word_probabilities
