import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
import string

dataframe_emails = pd.read_csv('emails.csv')
dataframe_emails.head()

print(f"Number of emails: {len(dataframe_emails)}")
print(f"Proportion of spam emails: {dataframe_emails.spam.sum()/len(dataframe_emails):.4f}")
print(f"Proportion of ham emails: {1-dataframe_emails.spam.sum()/len(dataframe_emails):.4f}")

def preprocess_emails(df):
    """
    Preprocesses email data from a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing email data with 'text' and 'spam' columns.

    Returns:
    - tuple: A tuple containing two elements:
        1. X (numpy.array): An array containing email content after removing the "Subject:" prefix.
        2. Y (numpy.array): An array indicating whether each email is spam (1) or ham (0).

    The function shuffles the input DataFrame to avoid biased results in train/test splits.
    It then extracts email content and spam labels, removing the "Subject:" prefix from each email.

    """
    # Shuffles the dataset
    df = df.sample(frac = 1, ignore_index = True, random_state = 42)
    # Removes the "Subject:" string, which comprises the first 9 characters of each email. Also, convert it to a numpy array.
    X = df.text.apply(lambda x: x[9:]).to_numpy()
    # Convert the labels to numpy array
    Y = df.spam.to_numpy()
    return X, Y

X, Y = preprocess_emails(dataframe_emails)

print(X[:5])
print(Y[:5])

email_index = 30
print(f"Email index {email_index}: {X[email_index]}\n\n")
print(f"Class: {Y[email_index]}")

def preprocess_text(X):
    """
    Preprocesses a collection of text data by removing stopwords and punctuation.

    Parameters:
    - X (str or array-like): The input text data to be processed. If a single string is provided,
      it will be converted into a one-element numpy array.

    Returns:
    - numpy.array: An array of preprocessed text data, where each element represents a document
      with stopwords and punctuation removed.

    Note:
    - The function uses the Natural Language Toolkit (nltk) library for tokenization and stopword removal.
    - If the input is a single string, it is converted into a one-element numpy array.
    """
    # Make a set with the stopwords and punctuation
    stop = set(stopwords.words('english') + list(string.punctuation))

    # The next lines will handle the case where a single email is passed instead of an array of emails.
    if isinstance(X, str):
        X = np.array([X])

    # The result will be stored in a list
    X_preprocessed = []

    for i, email in enumerate(X):
        email = np.array([i.lower() for i in word_tokenize(email) if i.lower() not in stop]).astype(X.dtype)
        X_preprocessed.append(email)
        
    if len(X) == 1:
        return X_preprocessed[0]
    return X_preprocessed

X_treated = preprocess_text(X)

email_index = 989
print(f"Email before preprocessing: {X[email_index]}")
print(f"Email after preprocessing: {X_treated[email_index]}")

TRAIN_SIZE = int(0.80*len(X_treated)) # 80% of the samples will be used to train.

X_train = X_treated[:TRAIN_SIZE]
Y_train = Y[:TRAIN_SIZE]
X_test = X_treated[TRAIN_SIZE:]
Y_test = Y[TRAIN_SIZE:]

print(f"Proportion of spam in train dataset: {sum(Y_train == 1)/len(Y_train):.4f}")
print(f"Proportion of spam in test dataset: {sum(Y_test == 1)/len(Y_test):.4f}")

def get_word_frequency(X,Y):
    """
    Calculate the frequency of each word in a set of emails categorized as spam (1) or not spam (0).

    Parameters:
    - X (numpy.array): Array of emails, where each email is represented as a list of words.
    - Y (numpy.array): Array of labels corresponding to each email in X. 1 indicates spam, 0 indicates ham.

    Returns:
    - word_dict (dict): A dictionary where keys are unique words found in the emails, and values
      are dictionaries containing the frequency of each word for spam (1) and not spam (0) emails.
    """
    # Creates an empty dictionary
    word_dict = {}


    num_emails = len(X)

    # Iterates over every processed email and its label
    for i in range(num_emails):
        # Get the i-th email
        email = X[i] 
        # Get the i-th label. This indicates whether the email is spam or not. 1 = None
        # The variable name cls is an abbreviation for class, a reserved word in Python.
        cls = Y[i] 
        # To avoid counting the same word twice in an email, remove duplicates by casting the email as a set
        email = set(email) 
        # Iterates over every distinct word in the email
        for word in email:
            # If the word is not already in the dictionary, manually add it. 
            if word not in word_dict.keys():
                word_dict[word] = {'spam':1, 'ham':1}
            # Add one occurrence for that specific word in the key ham if cls == 0 and spam if cls == 1. 
            if cls ==0:    
                word_dict[word]['ham'] += 1
            if cls ==1:
                word_dict[word]['spam'] += 1
    
    return word_dict

test_output = get_word_frequency([['like','going','river'], ['love', 'deep', 'river'], ['hate','river']], [1,0,0])
print(test_output)
        
word_frequency = get_word_frequency(X_train,Y_train)        
class_frequency = {'ham': sum(Y_train == 0), 'spam': sum(Y_train == 1)}
print(class_frequency)

proportion_spam = class_frequency['spam']/(class_frequency['ham'] + class_frequency['spam'])
print(f"The proportion of spam emails in training is: {proportion_spam:.4f}")

def prob_word_given_class(word, cls, word_frequency, class_frequency):
    """
    Calculate the conditional probability of a given word occurring in a specific class.

    Parameters:
    - word (str): The target word for which the probability is calculated.
    - cls (str): The class for which the probability is calculated, it may be 'spam' or 'ham'
    - word_frequency (dict): The dictionary containing the words frequency.
    - class_frequency (dict): The dictionary containing the class frequency.

    Returns:
    - float: The conditional probability of the given word occurring in the specified class.
    """
    
    # Get the amount of times the word appears with the given class (class is stores in spam variable)
    amount_word_and_class = word_frequency[word][cls]
    p_word_given_class = amount_word_and_class/class_frequency[cls]

    return p_word_given_class

print(f"P(lottery | spam) = {prob_word_given_class('lottery',cls='spam', word_frequency = word_frequency, class_frequency = class_frequency)}")
print(f"P(lottery | ham) = {prob_word_given_class('lottery', cls='ham', word_frequency = word_frequency, class_frequency = class_frequency)}")
print(f"P(schedule | spam) = {prob_word_given_class('schedule', cls='spam', word_frequency = word_frequency, class_frequency = class_frequency)}")
print(f"P(schedule | ham) = {prob_word_given_class('schedule', cls='ham', word_frequency = word_frequency, class_frequency = class_frequency)}") 

def prob_email_given_class(treated_email, cls, word_frequency, class_frequency):
    """
    Calculate the probability of an email being of a certain class (e.g., spam or ham) based on treated email content.

    Parameters:
    - treated_email (list): A list of treated words in the email.
    - cls (str): The class label for the email. It can be either 'spam' or 'ham'
    - word_frequency (dict): The dictionary containing the words frequency.
    - class_frequency (dict): The dictionary containing the class frequency.

    Returns:
    - float: The probability of the given email belonging to the specified class.
    """

    # prob starts at 1 because it will be updated by multiplying it with the current P(word | class) in every iteration
    prob = 1


    total_words_in_class = sum(word_frequency[word][cls] for word in word_frequency if cls in word_frequency[word])

    for word in treated_email:
        # Only perform the computation for words that exist in the word frequency dictionary
        if word in word_frequency.keys(): 
            # Update the prob by multiplying it with P(word | class). Don't forget to add the word_frequency and class_frequency parameters!
            prob *=(word_frequency[word].get(cls, 0) + 1) / (total_words_in_class + len(word_frequency))


    return prob

example_email = "Click here to win a lottery ticket and claim your prize!"
treated_email = preprocess_text(example_email)
prob_spam = prob_email_given_class(treated_email, cls = 'spam', word_frequency = word_frequency, class_frequency = class_frequency)
prob_ham = prob_email_given_class(treated_email, cls = 'ham', word_frequency = word_frequency, class_frequency = class_frequency)
print(f"Email: {example_email}\nEmail after preprocessing: {treated_email}\nP(email | spam) = {prob_spam}\nP(email | ham) = {prob_ham}")
   
   
def naive_bayes(treated_email, word_frequency, class_frequency, return_likelihood=False):
    """
    Naive Bayes classifier for spam detection.

    This function calculates the probability of an email being spam (1) or ham (0)
    based on the Naive Bayes algorithm. It uses the conditional probabilities of the
    treated_email given spam and ham, as well as the prior probabilities of spam and ham
    classes. The final decision is made by comparing the calculated probabilities.

    Parameters:
    - treated_email (list): A preprocessed representation of the input email.
    - word_frequency (dict): The dictionary containing the words frequency.
    - class_frequency (dict): The dictionary containing the class frequency.
    - return_likelihood (bool): If true, it returns the likelihood of both spam and ham.

    Returns:
    If return_likelihood = False:
        - int: 1 if the email is classified as spam, 0 if classified as ham.
    If return_likelihood = True:
        - tuple: A tuple with the format (spam_likelihood, ham_likelihood)
    """

    
    def prob_email_given_class(treated_email, cls, word_frequency, class_frequency):
        """
        Calculate the probability of an email being of a certain class (e.g., spam or ham) based on treated email content.
        """
        # prob starts at 1 because it will be updated by multiplying it with the current P(word | class) in every iteration
        prob = 1.0

        # Total number of words in the specified class
        total_words_in_class = sum(word_frequency[word][cls] for word in word_frequency if cls in word_frequency[word])
        
        for word in treated_email:
            # Only perform the computation for words that exist in the word frequency dictionary
            if word in word_frequency:
                # Update the prob by multiplying it with P(word | class)
                #P(word | cls) = (frequency of the word in the class + 1) / (total words in class + total unique words)
                prob *= (word_frequency[word].get(cls, 0) + 1) / (total_words_in_class + len(word_frequency))

        return prob

    # Compute P(email | spam) with the function you defined just above. Don't forget to add the word_frequency and class_frequency parameters!
    prob_email_given_spam = prob_email_given_class(treated_email, "spam", word_frequency, class_frequency)

    # Compute P(email | ham) with the function you defined just above. Don't forget to add the word_frequency and class_frequency parameters!
    prob_email_given_ham = prob_email_given_class(treated_email, "ham", word_frequency, class_frequency)

    # Compute P(spam) using the class_frequency dictionary and using the formula #spam emails / #total emails
    p_spam = class_frequency["spam"] / (class_frequency["spam"] + class_frequency["ham"])

    # Compute P(ham) using the class_frequency dictionary and using the formula #ham emails / #total emails
    p_ham = class_frequency["ham"] / (class_frequency["spam"] + class_frequency["ham"])

    # Compute the quantity P(spam) * P(email | spam), let's call it spam_likelihood
    spam_likelihood = p_spam * prob_email_given_spam

    # Compute the quantity P(ham) * P(email | ham), let's call it ham_likelihood
    ham_likelihood = p_ham * prob_email_given_ham

    # In case of passing return_likelihood = True, then return the desired tuple
    #f return_likelihood == True:
        #return (spam_likelihood, ham_likelihood)
    
    # Compares both values and choose the class corresponding to the higher value
    if spam_likelihood > ham_likelihood:
        return 1
    else:
        return 0
    
example_email = "Click here to win a lottery ticket and claim your prize!"
treated_email = preprocess_text(example_email)

print(f"Email: {example_email}\nEmail after preprocessing: {treated_email}\nNaive Bayes predicts this email as: {naive_bayes(treated_email, word_frequency, class_frequency)}")

print("\n\n")
example_email = "Our meeting will happen in the main office. Please be there in time."
treated_email = preprocess_text(example_email)

print(f"Email: {example_email}\nEmail after preprocessing: {treated_email}\nNaive Bayes predicts this email as: {naive_bayes(treated_email, word_frequency, class_frequency)}")

def get_true_positives(Y_true, Y_pred):
    """
    Calculate the number of true positive instances in binary classification.

    Parameters:
    - Y_true (list): List of true labels (0 or 1) for each instance.
    - Y_pred (list): List of predicted labels (0 or 1) for each instance.

    Returns:
    - int: Number of true positives, where true label and predicted label are both 1.
    """
    # Both Y_true and Y_pred must match in length.
    if len(Y_true) != len(Y_pred):
        return "Number of true labels and predict labels must match!"
    n = len(Y_true)
    true_positives = 0
    # Iterate over the number of elements in the list
    for i in range(n):
        # Get the true label for the considered email
        true_label_i = Y_true[i]
        # Get the predicted (model output) for the considered email
        predicted_label_i = Y_pred[i]
        # Increase the counter by 1 only if true_label_i = 1 and predicted_label_i = 1 (true positives)
        if true_label_i == 1 and predicted_label_i == 1:
            true_positives += 1
    return true_positives
        
def get_true_negatives(Y_true, Y_pred):
    """
    Calculate the number of true negative instances in binary classification.

    Parameters:
    - Y_true (list): List of true labels (0 or 1) for each instance.
    - Y_pred (list): List of predicted labels (0 or 1) for each instance.

    Returns:
    - int: Number of true negatives, where true label and predicted label are both 0.
    """
    
    # Both Y_true and Y_pred must match in length.
    if len(Y_true) != len(Y_pred):
        return "Number of true labels and predict labels must match!"
    n = len(Y_true)
    true_negatives = 0
    # Iterate over the number of elements in the list
    for i in range(n):
        # Get the true label for the considered email
        true_label_i = Y_true[i]
        # Get the predicted (model output) for the considered email
        predicted_label_i = Y_pred[i]
        # Increase the counter by 1 only if true_label_i = 0 and predicted_label_i = 0 (true negatives)
        if true_label_i == 0 and predicted_label_i == 0:
            true_negatives += 1
    return true_negatives
        
# Create an empty list to store the predictions
Y_pred = []


# Iterate over every email in the test set
for email in X_test:
    # Perform prediction
    prediction = naive_bayes(email, word_frequency, class_frequency)
    # Add it to the list 
    Y_pred.append(prediction)

# Checking if both Y_pred and Y_test (these are the true labels) match in length:
print(f"Y_test and Y_pred matches in length? Answer: {len(Y_pred) == len(Y_test)}")

# Get the number of true positives:
true_positives = get_true_positives(Y_test, Y_pred)

# Get the number of true negatives:
true_negatives = get_true_negatives(Y_test, Y_pred)

print(f"The number of true positives is: {true_positives}\nThe number of true negatives is: {true_negatives}")

# Compute the accuracy by summing true negatives with true positives and dividing it by the total number of elements in the dataset. 
# Since both Y_pred and Y_test have the same length, it does not matter which one you use.
accuracy = (true_positives + true_negatives)/len(Y_test)

print(f"Accuracy is: {accuracy:.4f}")

email = "Please meet me in 2 hours in the main building. I have an important task for you."
# email = "You win a lottery prize! Congratulations! Click here to claim it"

# Preprocess the email
treated_email = preprocess_text(email)
# Get the prediction, in order to print it nicely, if the output is 1 then the prediction will be written as "spam" otherwise "ham".
prediction = "spam" if naive_bayes(treated_email, word_frequency, class_frequency) == 1 else "ham"
print(f"The email is: {email}\nThe model predicts it as {prediction}.")

example_index = 4798
example_email = X[example_index]
treated_email = preprocess_text(example_email)
print(f"The email is:\n\t{example_email}\n\nAfter preprocessing:\n\t:{treated_email}")

spam_likelihood, ham_likelihood = naive_bayes(treated_email, word_frequency = word_frequency, class_frequency = class_frequency, return_likelihood = True)
print(f"spam_likelihood: {spam_likelihood}\nham_likelihood: {ham_likelihood}")

print(f"The example email is labeled as: {Y[example_index]}")
print(f"Naive bayes model classifies it as: {naive_bayes(treated_email, word_frequency, class_frequency)}")

print(f"The example email has: {len(treated_email)} words in the product.")

for i in range(3):
    word = treated_email[i]
    p_word_given_ham = prob_word_given_class(word, cls = 'ham', word_frequency = word_frequency, class_frequency = class_frequency)
    print(f"Word: {word}. P({word} | ham) = {p_word_given_ham}")
    
import sys

print(sys.float_info)

def log_prob_email_given_class(treated_email, cls, word_frequency, class_frequency):
    """
    Calculate the log probability of an email being of a certain class (e.g., spam or ham) based on treated email content.

    Parameters:
    - treated_email (list): A list of treated words in the email.
    - cls (str): The class label ('spam' or 'ham')
    

    Returns:
    - float: The log probability of the given email belonging to the specified class.
    """

    # prob starts at 0 because it will be updated by summing it with the current log(P(word | class)) in every iteration
    prob = 0

    for word in treated_email: 
        # Only perform the computation for words that exist in the word frequency dictionary
        if word in word_frequency.keys(): 
            # Update the prob by summing it with log(P(word | class))
            prob += np.log(prob_word_given_class(word, cls,word_frequency, class_frequency))

    return prob

# Consider an email with only one word, so it reduces to compute the value P(word | class) or log(P(word | class)).
one_word_email = ['schedule']
word = one_word_email[0]
prob_spam = prob_email_given_class(one_word_email, cls = 'spam',word_frequency = word_frequency, class_frequency = class_frequency)
log_prob_spam = log_prob_email_given_class(one_word_email, cls = 'spam',word_frequency = word_frequency, class_frequency = class_frequency)
print(f"For word {word}:\n\tP({word} | spam) = {prob_spam}\n\tlog(P({word} | spam)) = {log_prob_spam}")            

def log_naive_bayes(treated_email, word_frequency, class_frequency, return_likelihood = False):    
    """
    Naive Bayes classifier for spam detection, comparing the log probabilities instead of the actual probabilities.

    This function calculates the log probability of an email being spam (1) or ham (0)
    based on the Naive Bayes algorithm. It uses the conditional probabilities of the
    treated_email given spam and ham, as well as the prior probabilities of spam and ham
    classes. The final decision is made by comparing the calculated probabilities.

    Parameters:
    - treated_email (list): A preprocessed representation of the input email.
    - return_likelihood (bool): If true, it returns the log_likelihood of both spam and ham.

    Returns:
    - int: 1 if the email is classified as spam, 0 if classified as ham.
    """
    
    # Compute P(email | spam) with the new log function
    log_prob_email_given_spam = log_prob_email_given_class(treated_email, cls = 'spam',word_frequency = word_frequency, class_frequency = class_frequency) 

    # Compute P(email | ham) with the function you defined just above
    log_prob_email_given_ham = log_prob_email_given_class(treated_email, cls = 'ham',word_frequency = word_frequency, class_frequency = class_frequency) 

    # Compute P(spam) using the class_frequency dictionary and using the formula #spam emails / #total emails
    p_spam = class_frequency['spam']/(class_frequency['ham'] + class_frequency['spam']) 

    # Compute P(ham) using the class_frequency dictionary and using the formula #ham emails / #total emails
    p_ham = class_frequency['ham']/(class_frequency['ham'] + class_frequency['spam']) 

    # Compute the quantity log(P(spam)) + log(P(email | spam)), let's call it log_spam_likelihood
    log_spam_likelihood = np.log(p_spam) + log_prob_email_given_spam 

    # Compute the quantity P(ham) * P(email | ham), let's call it ham_likelihood
    log_ham_likelihood = np.log(p_ham) + log_prob_email_given_ham 

    # In case of passing return_likelihood = True, then return the desired tuple
    if return_likelihood == True:
        return (log_spam_likelihood, log_ham_likelihood)
    
    # Compares both values and choose the class corresponding to the higher value. 
    # As the logarithm is an increasing function, the class with the higher value retains this property.
    if log_spam_likelihood >= log_ham_likelihood:
        return 1
    else:
        return 0
    

log_spam_likelihood, log_ham_likelihood = log_naive_bayes(treated_email,word_frequency = word_frequency, class_frequency = class_frequency,return_likelihood = True)
print(f"log_spam_likelihood: {log_spam_likelihood}\nlog_ham_likelihood: {log_ham_likelihood}")    

print(f"The example email is labeled as: {Y[example_index]}")
print(f"Log Naive bayes model classifies it as: {log_naive_bayes(treated_email,word_frequency = word_frequency, class_frequency = class_frequency)}")


# Let's get the predictions for the test set:

# Create an empty list to store the predictions
Y_pred = []


# Iterate over every email in the test set
for email in X_test:
    # Perform prediction
    prediction = log_naive_bayes(email,word_frequency = word_frequency, class_frequency = class_frequency)
    # Add it to the list 
    Y_pred.append(prediction)

# Get the number of true positives:
true_positives = get_true_positives(Y_test, Y_pred)

# Get the number of true negatives:
true_negatives = get_true_negatives(Y_test, Y_pred)

print(f"The number of true positives is: {true_positives}\nThe number of true negatives is: {true_negatives}")

# Compute the accuracy by summing true negatives with true positives and dividing it by the total number of elements in the dataset. 
# Since both Y_pred and Y_test have the same length, it does not matter which one you use.
accuracy = (true_positives + true_negatives)/len(Y_test)

print(f"The accuracy is: {accuracy:.4f}")


def get_recall(Y_true, Y_pred):
    """
    Calculate the recall for a binary classification task.

    Parameters:
    - Y_true (array-like): Ground truth labels.
    - Y_pred (array-like): Predicted labels.

    Returns:
    - recall (float): The recall score, which is the ratio of true positives to the total number of actual positives.
    """
    # Get the total number of spam emails. Since they are 1 in the data, it suffices summing all the values in the array Y.
    total_number_spams = Y_test.sum()
    # Get the true positives
    true_positives = get_true_positives(Y_true, Y_pred)
    
    # Compute the recall
    recall = true_positives/total_number_spams
    return recall


def get_recall(Y_true, Y_pred):
    """
    Calculate the recall for a binary classification task.

    Parameters:
    - Y_true (array-like): Ground truth labels.
    - Y_pred (array-like): Predicted labels.

    Returns:
    - recall (float): The recall score, which is the ratio of true positives to the total number of actual positives.
    """
    # Get the total number of spam emails. Since they are 1 in the data, it suffices summing all the values in the array Y.
    total_number_spams = Y_test.sum()
    # Get the true positives
    true_positives = get_true_positives(Y_true, Y_pred)
    
    # Compute the recall
    recall = true_positives/total_number_spams
    return recall

print(f"The proportion of spam emails the standard Naive Bayes model can correctly classify as spam (recall) is: {recall_naive_bayes:.4f}")
print(f"The proportion of spam emails the log Naive Bayes model can correctly classify as spam (recall) is: {recall_log_naive_bayes:.4f}")

def get_false_positives(Y_true, Y_pred):
    """
    Calculate the number of false positives instances in binary classification.

    Parameters:
    - Y_true (list): List of true labels (0 or 1) for each instance.
    - Y_pred (list): List of predicted labels (0 or 1) for each instance.

    Returns:
    - int: Number of false positives, where true label is 0 and predicted label is 1.
    """
    
    # Both Y_true and Y_pred must match in length.
    if len(Y_true) != len(Y_pred):
        return "Number of true labels and predict labels must match!"
    n = len(Y_true)

    false_positives = 0
    # Iterate over the number of elements in the list
    for i in range(n):
        # Get the true label for the considered email
        true_label_i = Y_true[i]
        # Get the predicted (model output) for the considered email
        predicted_label_i = Y_pred[i]
        # Increase the counter by 1 only if true_label_i = 0 and predicted_label_i = 0 (false positive)
        if true_label_i == 0 and predicted_label_i == 1:
            false_positives += 1
    return false_positives


false_positives_naive_bayes = get_false_positives(Y_test, Y_pred_naive_bayes)
false_positives_log_naive_bayes = get_false_positives(Y_test, Y_pred_log_naive_bayes)


print(f"Number of false positives in the standard Naive Bayes model: {false_positives_naive_bayes}")
print(f"Number of false positives in the log Naive Bayes model: {false_positives_log_naive_bayes}")


def get_precision(Y_true, Y_pred):
    """
    Calculate precision, a metric for the performance of a classification model,
    by computing the ratio of true positives to the sum of true positives and false positives.

    Parameters:
    - Y_true (list): True labels.
    - Y_pred (list): Predicted labels.

    Returns:
    - precision (float): Precision score.
    """
    # Get the true positives
    true_positives = get_true_positives(Y_true, Y_pred)
    false_positives = get_false_positives(Y_true, Y_pred)
    precision = true_positives/(true_positives + false_positives)
    return precision

print(f"Precision of the standard Naive Bayes model: {get_precision(Y_test, Y_pred_naive_bayes):.4f}")
print(f"Precision of the log Naive Bayes model: {get_precision(Y_test, Y_pred_log_naive_bayes):.4f}")


    """
    The first version of the model has a precision of 59.57%. In other words, from 100 emails the model classifies as spam, 
    only around 60 of them are in fact spam. This means that this model would send 40 ham emails to the spam folder, 
    indicating that, even though very sensitive, it is not very reliable. 
    On the other hand, the improved model has a precision of 98.42%! So from 100 emails classified as spam,
    only around 2 will be actually ham emails. A much more reliable output. 
    """
