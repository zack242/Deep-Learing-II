import os 
import requests 
import scipy.io as sio
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from scipy.special import expit


########## Partie 2 - Data loading 
def fetch_alpha_digits_data():
    alphadigs_url = 'https://cs.nyu.edu/~roweis/data/binaryalphadigs.mat'
    r = requests.get(alphadigs_url, allow_redirects=True)
    filename = 'binaryalphadigs.mat'
    open('data/' + filename, 'wb').write(r.content)
    
def fetch_mnist_digits_data(data_length):
    # Fetch data 
    mnist = fetch_openml('mnist_784')

    # Get pixels values
    X = mnist['data'].values.copy() #apply(lambda x: 1 if x > 125 else 0)
    # From grayscale to black and white
    X[X < 125] = 0 
    X[X >= 125] = 1


    #train_length = (int)(data_length*test_train)

    X_train, X_test = X[:data_length, :], X[data_length:data_length+10000, :]
    # get labels 
    y = mnist['target'].values.copy()
    y_train, y_test = y[:data_length], y[data_length:data_length+10000]
    # One hot encoding for labels 
    y_train = LabelBinarizer().fit_transform(y_train)
    y_test = LabelBinarizer().fit_transform(y_test)
    return X_train, X_test, y_train, y_test
    
def lire_alpha_digit(digits_list):
    """Read alpha_digit data from specific digits_list

    Keyword arguments:
    digits_list -- list of digits to be fetched as str
    ['0', '2', 'F', 'Z']
    
    Return: 
    alpha digits data regarding the input digits as a np.array matrix
    (row is a data point, columns are the features of each picture )
    """
    # check if data is in data/ folder 
    if os.path.exists('data/binaryalphadigs.mat'):
        print('File already downloaded, using version in data folder..')
    # download it if necessary
    else:
        print('Fetching data on internet...')
        fetch_alpha_digits_data()
    # load data 
    alphadigs_dict = sio.loadmat('data/binaryalphadigs.mat')
    
    # filter digits
    digit2idx = {}
    for i, digit in enumerate(alphadigs_dict['classlabels'][0]):
        digit2idx[digit[0]] = i
    
    # collect indexes 
    idxs = []
    for digit in digits_list:
        idxs.append(digit2idx[digit])
    #return alphadigs_dict['dat'][idxs]
    # Adapter au format (n, p), chaque colonne designe un pixel et chaque ligne une image
    return np.stack(np.concatenate(alphadigs_dict['dat'][idxs])).reshape(-1, 20*16)

def sigmoid(x):
    return expit(x)
    
