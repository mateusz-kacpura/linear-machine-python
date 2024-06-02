import numpy as np
from numpy.random import seed
import skimage.io
import skimage.transform
import os
import random
from skimage.color import rgb2gray
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QGridLayout, QVBoxLayout, QMessageBox, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt
import sys
import matplotlib.pyplot as plt
import pickle

TRAINING_DATA_PNG = r"C:\Users\engli\Sieci neuronowe\linear-machine-python\images"

'''
    images/
    ├── 0/
    │   ├── d4bdebc2-fcc6-4d60-904f-7eb289cc9a7d.png
    │   └── 4f23bb71-1c45-4d61-a7de-2d2de724f274.png
    ├── 1/
    │   ├── 03d22b11-6096-4d53-8ff1-ef7a0d42c18a.png
    │   ├── c1995d1a-fa52-4792-8d11-8875f24558e6.png
    │   └── 9b52c040-2023-40e6-9861-8bc2197d17de.png
    ├── 2/
    │   ├── e8f88c9e-90e8-43c3-b833-d1e71d5a46c8.png
    │   └── 6fc169b4-78d7-4b48-8e63-f4b8394c2a14.png
    └── ...
'''

DEBUGGING = 'ON' ## ON / OFF

## Grid (px)
GRID_WIDTH = 20
GRID_HEIGHT = 20
CELL_SIZE = 25
IMAGE_SIZE = 20

# Noise <0,1>
NOISE_LEVEL_MIN = 0
NOISE_LEVEL_MAX = 0.1

## Linear perceptron  / backpropagation parametrs
NUM_CATEGORIES = 10
ITERATIONS = 10000
LEARNING_RATE = 0.1

## Training
TRAIN_NOISE = False
SHIFT = True
NUMBERS_SHIFTS = 2

## Photos are generated based on random offset values
PHOTO_MULTIPLER = True
NUMBERS_PHOTO_MULTIPLER = 3

## CNN
FILTER_SIZE = 3
STRIDE = 1
PADDING = 1

# Zmienne globalne przechowująca wytrenowany model
ADALINE_MODEL = []
LINEAR_MODEL = None
BACKPROPAGATION = None


class CNN:
    print ("Algorytm CNN nie został jeszcze zaimpolementowany")
    pass

class ModelHandler:
    def __init__(self):
        self.save_models.clicked.connect(self.event_save_models)
        self.load_models.clicked.connect(self.event_load_models)

    def event_save_models(self):
        models = {
            'ADALINE_MODEL': ADALINE_MODEL,
            'LINEAR_MODEL': LINEAR_MODEL,
            'BACKPROPAGATION': BACKPROPAGATION
        }
        
        with open('models.pkl', 'wb') as file:
            pickle.dump(models, file)
        print("Models saved successfully.")

    def event_load_models(self):
        global ADALINE_MODEL, LINEAR_MODEL, BACKPROPAGATION

        if os.path.exists('models.pkl'):
            with open('models.pkl', 'rb') as file:
                models = pickle.load(file)
                ADALINE_MODEL = models.get('ADALINE_MODEL', [])
                LINEAR_MODEL = models.get('LINEAR_MODEL', None)
                BACKPROPAGATION = models.get('BACKPROPAGATION', None)
            print("Models loaded successfully.")
        else:
            print("No saved models found.")

class AdalineSGD(object):

    """
    Parametry
    -----------
    eta : float
        Współczynnik uczenia (pomiędzy 0.0 a 1.0)
    n_iter : int
        Ilość przejść przez zbiór treningowy.

    Atrybuty
    -----------
    w_ : 1d-array
        Wagi po dopasowaniu.
    errors_ : lista
        Liczba błędnych klasyfikacji w każdej epoce.
    shuffle : bool (domyślnie: True)
        Miesza dane treningowe w każdej epoce, jeśli True, 
        aby zapobiec cyklom.
    random_state : int (domyślnie: None)
        Ustawia stan losowy do mieszania i 
        inicjalizowania wag.

    """

    def __init__(self, eta = 0.01, n_iter = 10, shuffle= True,
                random_state = None):

        self.eta = eta
        self.n_iter = n_iter
        self.w_initialization = False
        self.shuffle = shuffle

        if random_state:
            seed(random_state)

    def fit(self, X, y):

        """ 	
        Parametry
        ------------
        X : {array-like}, kształt = [n_samples, n_features]
            Wektory treningowe, gdzie n_samples to
            liczba próbek, a n_features to liczba
            cech.
        y : array-like, kształt = [n_samples]
            Wartości docelowe.

        Zwróć
        -------
        self : obiekt

        """


        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):

            if self.shuffle:
                X, y = self._shuffle(X, y)

            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))

            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, y):

        """ Fit training data without reinitializing the weights """

        if not self.w_initialized:
            self._initialize_weights(X.shape[1])

        if y.ravel().shape[0] > 1:

            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)

        return self

    def _shuffle(self, X, y):

        """ Shuffle training data """

        r = np.random.permutation(len(y))

        return X[r], y[r]

    def _initialize_weights(self, m):

        """ Initialize weights to zeros """

        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):

        """ Apply Adaline learning rule to update the weights """

        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * (error ** 2)

        return cost

    def net_input(self, X):

        """ Calculate net input """

        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):

        """ Compute linear activation """

        return self.net_input(X)

    def predict(self, X):

        """ Return class label after the unit step """

        return np.where(self.activation(X) >= 0.0, 1, -1)
    
    def plot_learning_curve(self):
        """ Plot the learning curve of the AdalineSGD """
        plt.plot(range(1, len(self.cost_) + 1), self.cost_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Average Cost')
        plt.title('Adaline - Learning rate {}'.format(self.eta))
        plt.show()

class Neuron:
    def __init__(self, number_of_inputs, eta=0.1):
        self.number_of_inputs = number_of_inputs
        self.eta = eta
        self.delta = 0
        self.weights = np.random.random(number_of_inputs) - 0.5
        self.bias = np.random.random() - 0.5
        self.output = 0
        self.preactivation = 0
        self.input = 0

    def activation(self, x):
        return 1. / (1 + np.exp(-x))

    def activation_derivative(self):
        return self.output * (1 - self.output)

    def predict(self, x):
        self.input = x.copy()
        self.preactivation = np.dot(self.weights, x) + self.bias
        self.output = self.activation(self.preactivation)
        return self.output

    def update_weights(self):
        for i in range(self.number_of_inputs):
            self.weights[i] -= self.eta * self.delta * self.input[i]
        self.bias -= self.eta * self.delta

class Layer:
    def __init__(self, layer_size, prevlayer_size, eta=0.1):
        self.layer_size = layer_size
        self.prevlayer_size = prevlayer_size
        self.eta = eta
        self.neurons = [Neuron(prevlayer_size, eta) for _ in range(layer_size)]

    def predict(self, x):
        return [neuron.predict(x) for neuron in self.neurons]

    def update_weights(self):
        for neuron in self.neurons:
            neuron.update_weights()

class NeuralNetwork:
    def __init__(self, structure, eta, iterations=100):
        self.structure = structure
        self.eta = eta
        self.iterations = iterations
        self.network_size = len(structure) - 1
        self.layers = [Layer(structure[i + 1], structure[i], eta) for i in range(self.network_size)]
        self.errors = []

    def forward(self, x):
        inputs = x.copy()
        for layer in self.layers:
            inputs = layer.predict(inputs)
        self.output = np.array(inputs)
        return self.output

    def backward(self, y):
        y = np.array(y)
        last_layer = self.network_size - 1

        # Adjust for multiple outputs
        for j in range(self.layers[last_layer].layer_size):
            epsilon = self.output[j] - y[j]
            self.layers[last_layer].neurons[j].delta = epsilon * self.layers[last_layer].neurons[j].activation_derivative()

        for l in reversed(range(last_layer)):
            for j in range(self.layers[l].layer_size):
                epsilon = sum(self.layers[l + 1].neurons[k].delta * self.layers[l + 1].neurons[k].weights[j] for k in range(self.layers[l + 1].layer_size))
                self.layers[l].neurons[j].delta = epsilon * self.layers[l].neurons[j].activation_derivative()

        for layer in self.layers:
            layer.update_weights()

    def train(self, train_x, train_y):
        self.errors = []
        for _ in range(self.iterations):
            for x, y in zip(train_x, train_y):
                self.forward(x)
                self.backward(y)  # Adjusted to accept array of outputs
            self.errors.append(self.error(train_x, train_y))

    def error(self, train_x, train_y):
        e = 0
        for x, y in zip(train_x, train_y):
            o = self.forward(x)
            e += np.linalg.norm(o - y)
        return e / len(train_x)

    def plot_errors(self):
        plt.plot(self.errors)
        plt.title("Training Error Progress")
        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.show()

class LinearPerceptron():
    def __init__(self, no_of_inputs):
        self.no_of_inputs = no_of_inputs
        self.weights = np.random.random((self.no_of_inputs + 1))
    def output(self, x):
        x = np.append(x, 1)  # Bias
        return np.dot(self.weights, x)

class LinearMachine():
    def __init__(self, no_of_inputs, no_of_categories, iterations=ITERATIONS, eta=LEARNING_RATE):
        self.no_of_inputs = no_of_inputs
        self.iterations = iterations
        self.eta = eta
        self.no_of_categories = no_of_categories
        self.perceptrons = []
        for i in range(self.no_of_categories):
            self.perceptrons.append(LinearPerceptron(self.no_of_inputs))

    def output(self, x):
        out = np.zeros(self.no_of_categories)
        for i in range(self.no_of_categories):
            out[i] = self.perceptrons[i].output(x)
        return np.argmax(out)

    def train(self, X, y):
        for i in range(self.iterations):
            for x, l in zip(X, y):
                k = self.output(x)
                if k != l:
                    self.perceptrons[k].weights[1:] += self.eta * (x - self.perceptrons[k].weights[1:])
                    self.perceptrons[k].weights[0] += self.eta * (-1)
                    self.perceptrons[l].weights[1:] += self.eta * (x - self.perceptrons[l].weights[1:])
                    self.perceptrons[l].weights[0] += self.eta * (1)

        self.plot_training_process(i, X, y)

    def predict(self, x):
        return self.output(x)
    
    def plot_training_process(self, iteration, X, y):
        plt.figure(figsize=(8, 6))
        for idx, point in enumerate(X):
            if y[idx] == 0:
                plt.scatter(point[0], point[1], color='red', marker='o')
            elif y[idx] == 1:
                plt.scatter(point[0], point[1], color='blue', marker='x')
        
        x_values = [np.min(X[:, 0] - 1), np.max(X[:, 0] + 1)]
        for i, perceptron in enumerate(self.perceptrons):
            y_values = -(perceptron.weights[1] * np.array(x_values) + perceptron.weights[0]) / perceptron.weights[2]
            plt.plot(x_values, y_values, label=f'Perceptron {i + 1}')
        
        plt.title(f'Iteration {iteration + 1}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()

class Grid(QWidget):
    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT, cell_size=CELL_SIZE, add_noise_function=None):
        super().__init__()
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid = [[False for _ in range(width)] for _ in range(height)]
        self.noisy_grid = np.zeros((height, width))
        self.buttons = [[QPushButton(self) for _ in range(width)] for _ in range(height)]
        for row in range(height):
            for col in range(width):
                self.buttons[row][col].setStyleSheet(f"background-color: white; border: none;")
                self.buttons[row][col].setFixedSize(cell_size, cell_size)
                self.buttons[row][col].clicked.connect(self.make_toggle(row, col))
                self.buttons[row][col].enterEvent = lambda _, row=row, col=col: self.on_enter(row, col)
        self.layout = QGridLayout()
        self.layout.setSpacing(0)
        for row in range(height):
            for col in range(width):
                self.layout.addWidget(self.buttons[row][col], row, col)
        self.setLayout(self.layout)
        self.selection_mode = SelectionMode.NONE
        self.clear()
        self.add_noise_function = add_noise_function

    def make_toggle(self, row, col):
        def toggle():
            if self.selection_mode == SelectionMode.SELECT:
                self.grid[row][col] = True
                self.buttons[row][col].setStyleSheet("background-color: black;")
            elif self.selection_mode == SelectionMode.DESELECT:
                self.grid[row][col] = False
                self.buttons[row][col].setStyleSheet("background-color: white;")
            else:
                self.grid[row][col] = not self.grid[row][col]
                color = "black" if self.grid[row][col] else "white"
                self.buttons[row][col].setStyleSheet(f"background-color: {color};")
                print(f"Toggled cell ({row}, {col})")
        return toggle

    def on_enter(self, row, col):
        if self.selection_mode == SelectionMode.SELECT:
            self.grid[row][col] = True
            self.buttons[row][col].setStyleSheet("background-color: black;")
        elif self.selection_mode == SelectionMode.DESELECT:
            self.grid[row][col] = False
            self.buttons[row][col].setStyleSheet("background-color: white;")

    def add_noise_to_grid(self):
        for row in range(self.height):
            for col in range(self.width):
                image = self.get_image_from_grid()
                noisy_image = self.add_noise_function(image, NOISE_LEVEL_MAX)
                if self.grid[row][col] == 0:
                    self.noisy_grid[row][col] = noisy_image[row][col]
                    self.update_cell_from_image(self.noisy_grid, row, col)

    def get_noise_from_grid(self):
        image = np.zeros((self.height, self.width))
        for row in range(self.height):
            for col in range(self.width):
                if self.noisy_grid[row][col]:
                    image[row][col] = 1
        return image

    def get_image_from_grid(self):
        image = np.zeros((self.height, self.width))
        for row in range(self.height):
            for col in range(self.width):
                if self.grid[row][col]:
                    image[row][col] = 1
        return image

    def update_cell_from_image(self, image, row, col):
        color = "black" if image[row][col] else "white"
        self.buttons[row][col].setStyleSheet(f"background-color: {color};")

    def update(self, new_image):    
        self.clear()
        self.grid = new_image
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                self.update_cell_from_image(new_image, row, col)

    def clear(self):
        for row in range(self.height):
            for col in range(self.width):
                self.grid[row][col] = False
                self.noisy_grid[row][col] = 0
                self.buttons[row][col].setStyleSheet("background-color: white;")

class SelectionMode:
    NONE = 0
    SELECT = 1
    DESELECT = 2

class Interface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 600, 600)
        self.setWindowTitle("Sieci neuronowe")
        self.create_title_bar()
        self.grid = Grid(GRID_WIDTH, GRID_HEIGHT, CELL_SIZE, add_noise_function=self.add_noise)
        self.create_buttons()
        self.show()

    def create_title_bar(self):
        title_label = QLabel("Pixel Grid")
        title_font = title_label.font()
        title_font.setPointSize(20)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)

        title_layout = QVBoxLayout()
        title_layout.addWidget(title_label)

        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_layout.addLayout(title_layout)
        central_widget.setLayout(central_layout)

        self.setCentralWidget(central_widget)

    def create_buttons(self):
        self.save_button = QPushButton("Save picture (Ctr + s)")
        self.select_button = QPushButton("Select Mode (Ctr + a)")
        self.deselect_button = QPushButton("Deselect Mode (Ctr + q)")
        self.train_linear_button = QPushButton("Liner Machine training")
        self.AdalineSGD_button = QPushButton("Adaline")
        self.train_backpropagations = QPushButton("Backpropagation training")
        self.test_button = QPushButton("Test")
        self.clear_button = QPushButton("Clear (Ctr + x)")
        self.help_button = QPushButton("Help (Ctr + h)")
        self.save_models = QPushButton("Save models")
        self.load_models = QPushButton("Load models")
        self.move_left_button = QPushButton("← Move Left")
        self.move_right_button = QPushButton("→ Move Right")
        self.move_up_button = QPushButton("↑ Move Up")
        self.move_down_button = QPushButton("↓ Move Down")
        
        arrow_button_layout = QHBoxLayout()
        arrow_button_layout.addWidget(self.move_left_button)
        arrow_button_layout.addWidget(self.move_up_button)
        arrow_button_layout.addWidget(self.move_right_button)
        arrow_button_layout.addWidget(self.move_down_button)

        train_button_layout = QHBoxLayout()
        train_button_layout.addWidget(self.train_linear_button)
        train_button_layout.addWidget(self.AdalineSGD_button)
        train_button_layout.addWidget(self.train_backpropagations)

        test_button_layout = QVBoxLayout()
        test_button_layout.addWidget(self.test_button)

        manage_button_layout = QHBoxLayout()
        manage_button_layout.addWidget(self.select_button)
        manage_button_layout.addWidget(self.deselect_button)

        other_button_layout = QHBoxLayout()
        other_button_layout.addWidget(self.save_models)
        other_button_layout.addWidget(self.load_models)
        other_button_layout.addWidget(self.save_button)
        other_button_layout.addWidget(self.clear_button)

        help_button_layout = QVBoxLayout()
        help_button_layout.addWidget(self.help_button)

        button_layout = QVBoxLayout()
        button_layout.addLayout(arrow_button_layout)
        button_layout.addLayout(train_button_layout)
        button_layout.addLayout(test_button_layout)
        button_layout.addLayout(manage_button_layout)
        button_layout.addLayout(other_button_layout)
        button_layout.addLayout(help_button_layout)

        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_layout.addWidget(self.grid)
        central_layout.addLayout(button_layout)
        central_widget.setLayout(central_layout)
        
        self.setCentralWidget(central_widget)
        self.save_button.clicked.connect(self.create_save_buttons)
        self.select_button.clicked.connect(self.enable_select_mode)
        self.deselect_button.clicked.connect(self.enable_deselect_mode)
        self.train_linear_button.clicked.connect(self.train_linear_machine)
        self.AdalineSGD_button.clicked.connect(self.AdalineSGD_train)
        self.train_backpropagations.clicked.connect(self.train_backpropagation)
        self.test_button.clicked.connect(self.test)
        self.save_models.clicked.connect(ModelHandler.event_save_models)
        self.load_models.clicked.connect(ModelHandler.event_load_models)
        self.clear_button.clicked.connect(self.clear)
        self.help_button.clicked.connect(self.show_help)
        self.move_left_button.clicked.connect(self.move_left)
        self.move_right_button.clicked.connect(self.move_right)
        self.move_up_button.clicked.connect(self.move_up)
        self.move_down_button.clicked.connect(self.move_down)

    def shift_image(self, image, direction):
        image_2d = image.reshape((GRID_HEIGHT, GRID_WIDTH))  # Assuming HEIGHT and WIDTH are defined
        if direction == 'up':
            return np.roll(image_2d, -1, axis=0).reshape(-1)
        elif direction == 'down':
            return np.roll(image_2d, 1, axis=0).reshape(-1)
        elif direction == 'left':
            return np.roll(image_2d, -1, axis=1).reshape(-1)
        elif direction == 'right':
            return np.roll(image_2d, 1, axis=1).reshape(-1)
        else:
            raise ValueError("Invalid direction. Choose from 'up', 'down', 'left', 'right'.")

    def generate_shifted_images(self, temp_image_gray):
        directions = []

        for i in range(NUMBERS_SHIFTS):
            left = np.random.randint(0, 2)
            up = np.random.randint(0, 2)
            down = np.random.randint(0, 2)
            right = np.random.randint(0, 2)

            if left == 1:
                directions.append('left')
            if up == 1:
                directions.append('up')
            if down == 1:
                directions.append('down')
            if right == 1:
                directions.append('right')

        print (directions)
        shifted_image = temp_image_gray  # Initialize with original image
        for direction in directions:
            shifted_image = self.shift_image(shifted_image, direction)

        return shifted_image

    def train_test_split(self, x, y, test_size=0.2, random_state=None):
        """
        Podział danych na zbiory treningowe i testowe.

        Args:
            x (list or np.array): Tablica z danymi.
            y (list or np.array): Tablica z etykietami.
            test_size (float): Procent danych do użycia jako zbiór testowy (domyślnie 0.2).
            random_state (int): Ziarno losowości (domyślnie None).

        Returns:
            tuple: Krotka zawierająca dane treningowe i testowe: (x_train, x_test, y_train, y_test).
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Losowe indeksy danych do zbioru testowego
        test_indices = np.random.choice(len(x), size=int(len(x) * test_size), replace=False)
        train_indices = np.setdiff1d(np.arange(len(x)), test_indices)

        x_train = [x[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]

        x_test = [x[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]

        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    def preprocess_training_data(self):
        global NUMBERS_PHOTO_MULTIPLER
        global DEBUGGING

        print("Training...")

        training_data = []
        number_machine = []

        for i in range(NUM_CATEGORIES):
            subdir = os.path.join(TRAINING_DATA_PNG, str(i))
            try:
                for filename in os.listdir(subdir):
                    if filename.endswith(".png"):
                        filepath = os.path.join(subdir, filename)
                        temp_image = skimage.io.imread(filepath)
                        temp_image = skimage.transform.resize(temp_image, (IMAGE_SIZE, IMAGE_SIZE))
                        
                        if temp_image.ndim == 3: 
                            temp_image_gray = rgb2gray(temp_image)
                        else:
                            temp_image_gray = temp_image
                        
                        if (PHOTO_MULTIPLER == False):
                            NUMBERS_PHOTO_MULTIPLER = 1

                        for o in range(NUMBERS_PHOTO_MULTIPLER):
                            if (TRAIN_NOISE ==  True):
                                noise_level = np.random.uniform(NOISE_LEVEL_MIN, NOISE_LEVEL_MAX)
                                temp_image_noisy = self.add_noise(temp_image_gray, noise_level)
                                training_data.append(temp_image_noisy.reshape(-1))
                            else:
                                training_data.append(temp_image_gray.reshape(-1))
                                number_machine.append(i)

                            if (SHIFT == True):
                                shifted_image = self.generate_shifted_images(temp_image_gray)
                                training_data.append(shifted_image)
                                number_machine.append(i)

            except FileNotFoundError:
                print(f"Directory '{subdir}' not found.")
        
        '''
        if DEBUGGING == 'ON':
            for id, data in enumerate(training_data):
                print(f"Size of element {id}: {data.shape}")
        '''

        training_data = np.array(training_data)
        number_machine = np.array(number_machine)

        x_train, x_test, y_train, y_test = self.train_test_split(training_data, number_machine, test_size=0.2, random_state=42)
        
        print("training_data:", training_data.shape)
        print("number_machine:", number_machine.shape)
        print("x_train:", x_train.shape)
        print("x_test:", x_test.shape)
        print("y_train:", y_train.shape)
        print("y_test:", y_test.shape)

        y_train_unique = np.unique(y_train)
        if len(y_train_unique) != NUM_CATEGORIES:
            print("Liczba kategorii musi być równa licznie unikalnych wartości y zbioru treningowego")
            print("y_train to", y_train)
            print("NUM_CATEGORIES to", NUM_CATEGORIES)

        return x_train, x_test, y_train, y_test

    def train_linear_machine(self):
        global LINEAR_MODEL  # Dodajemy globalne odwołanie

        x_train, x_test, y_train, y_test = self.preprocess_training_data()

        N = GRID_WIDTH

        # Inicjalizacja maszyny liniowej, jeśli jeszcze nie istnieje
        if LINEAR_MODEL is None:
            LINEAR_MODEL = LinearMachine(N * N, NUM_CATEGORIES)

        # Trening maszyny liniowej
        y = [y_train[index] for index in range(len(x_train))]
        LINEAR_MODEL.train(x_train, y)

        # Dokładność na danych treningowych
        total_correct = sum(1 for x, y_true in zip(x_train, y_train) if LINEAR_MODEL.predict(x) == y_true)
        total_samples = len(y_train)
        overall_accuracy = total_correct / total_samples
        print(f"Overall accuracy on training data: {overall_accuracy}")

    def AdalineSGD_train(self):
        global ADALINE_MODEL
        x_train, x_test, y_train, y_test= self.preprocess_training_data()

        """
        x_train przechowuje macieże np jednowymiarowe 400
        y_train przechowuje opisy kategorii
        """

        adaline_models = []

        for i in range (0,NUM_CATEGORIES):
            adaline = AdalineSGD(eta=LEARNING_RATE, n_iter=ITERATIONS)
            adaline.fit(x_train, y_train)  # Dopasowanie modelu
            adaline_models.append(adaline)
            adaline.plot_learning_curve()
            
        ADALINE_MODEL = adaline_models

    def one_hot_encode(self, y_train, num_classes):
        one_hot_encoded = np.zeros((len(y_train), num_classes))
        
        for idx, value in enumerate(y_train):
            one_hot_encoded[idx, value] = 1
        
        return one_hot_encoded.tolist()

    def train_backpropagation(self):

        global BACKPROPAGATION

        x_train, x_test, y_train, y_test= self.preprocess_training_data()
        
        N = GRID_WIDTH
        
        one_hot_encoded_y_train = self.one_hot_encode(y_train, NUM_CATEGORIES)
        
        '''
        if DEBUGGING == 'ON':
            print("Dane y_train zostały poprawnie przekonwertowane do postaci one-hot encoded vectors)
            ### Tylko takie dane przyjmuje algorytm wstecznej propagacji błędu
            print("Wymiar przekonweerownych danych:", one_hot_encoded_y_train.shape())
        '''

        adaline_models = []
        structure = [N * N, 2, 10]
        neural_network = NeuralNetwork(structure, eta=LEARNING_RATE, iterations=ITERATIONS)

        neural_network.train(x_train, one_hot_encoded_y_train)
        
        test_error = neural_network.error(x_test, y_test)
        print("Test error:", test_error)

        neural_network.plot_errors()
        BACKPROPAGATION = neural_network

    def check_category(self, flattened_image):
        activations = []
        for model in ADALINE_MODEL:
            activations.append(model.activation(flattened_image))

        most_activated_index = np.argmax(activations)
        return most_activated_index

    def test(self):
        global ADALINE_MODEL
        global LINEAR_MODEL
        image_data = self.grid.get_image_from_grid() ## funkcja zwraca prawidlowo wartości w formacie macieży z wartościami 0 i 1
        noisy_image_data = self.grid.get_noise_from_grid() ## funkcja jest do zaimplementowania powinna zwracać tablice w formacie grid z wartościami 0 i 1
        merged_image = image_data + noisy_image_data
        flattened_image = merged_image.flatten()

        if ADALINE_MODEL is None:
            print("Model Adaline nie jest wytrenowany. Proszę najpierw go wytrenować.")
        else:
            category = self.check_category(flattened_image)
            print ("Algorytm adline rozpoznał liczbę", category)

        if LINEAR_MODEL is None:
            print("Model maszyny liniowej nie jest wytrenowany. Proszę najpierw go wytrenować.")
        else:
            category = LINEAR_MODEL.predict(merged_image)
            print ("Maszyna liniowa rozpoznała liczbę: ", category)

        if BACKPROPAGATION is None:
            print("Model propagacji wstecznej nie jest wytrenowany. Proszę najpierw go wytrenować.")
        else:
            print(f"Flattened image shape: {flattened_image.shape}")

            predictions = BACKPROPAGATION.forward(flattened_image)
            print(f"Predictions: {predictions}")
            if predictions.size == 0:
                raise ValueError("Predictions array is empty")
            predict_class = np.argmax(predictions)
            print(f"Wsteczna propagacja rozmoznała liczbę: {predict_class}")

    def pearsonr(self, x, y):
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        cov_xy = np.mean((x - mean_x) * (y - mean_y))
        std_x = np.std(x)
        std_y = np.std(y)

        if std_x != 0 and std_y != 0:
            correlation = cov_xy / (std_x * std_y)
        else:
            correlation = 0.0

        return correlation

    def calculate_correlation(self):

        data = []
        number_machine = []

        for i in range(NUM_CATEGORIES):
            subdir = os.path.join(TRAINING_DATA_PNG, str(i))
            try:
                for filename in os.listdir(subdir):
                    if filename.endswith(".png"):
                        filepath = os.path.join(subdir, filename)
                        temp_image = skimage.io.imread(filepath)
                        temp_image = skimage.transform.resize(temp_image, (IMAGE_SIZE, IMAGE_SIZE))
                        data.append(temp_image.flatten())
                        number_machine.append(i)
            except FileNotFoundError:
                print(f"Directory '{subdir}' not found.")

        data = np.array(data)
        number_machine = np.array(number_machine)

        num_samples, num_features = data.shape
        correlation_matrix = np.zeros((num_features, num_features))
        for i in range(num_features):
            for j in range(num_features):
                correlation_matrix[i, j] = self.pearsonr(data[:, i], data[:, j])
        
        print("Wymiary macierzy:")
        print(correlation_matrix.shape)
        
        # Wypisanie zawartości macierzy
        print("\nZawartość macierzy korelacji:")
        print(correlation_matrix)

        return correlation_matrix

    def add_noise(self, image, noise_level):
        noise = np.random.binomial(1, noise_level, size=image.shape)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 1)
        return noisy_image

    def denoise(self, image):
        return skimage.restoration.denoise_tv_chambolle(image)
    
    def add_noise_to_grid(self):
        self.grid.add_noise_to_grid()

    def clear(self):
        self.grid.clear()

    def enable_select_mode(self):
        self.grid.selection_mode = SelectionMode.SELECT

    def enable_deselect_mode(self):
        self.grid.selection_mode = SelectionMode.DESELECT

    def create_save_buttons(self):
        button_layout = QVBoxLayout()
        for i in range(NUM_CATEGORIES):
            save_button = QPushButton(f"Save {i} ( {i})")
            save_button.clicked.connect(lambda _, idx=i: self.save(idx))
            button_layout.addWidget(save_button)

        backspace_button = QPushButton("Backspace <-")
        backspace_button.clicked.connect(self.back_to_paint_mode)
        button_layout.addWidget(backspace_button)

        central_widget = QWidget()
        central_widget.setLayout(button_layout)
        self.setCentralWidget(central_widget)

    def generate_uuid4(self):
        time_part = format(random.getrandbits(32), '08x')
        clock_seq = format(random.getrandbits(16), '04x')
        version = "4"
        variant = format(random.randint(8, 11), '01x')
        random_part = ''.join(format(random.getrandbits(8), '02x') for _ in range(12))
        uuid_str = '-'.join((time_part, clock_seq, version + variant, random_part))
        return uuid_str

    def save(self, folder_index):
        directory = f"images/{folder_index}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = str(self.generate_uuid4()) + ".png"
        filepath = os.path.join(directory, filename)

        image_data = self.grid.get_image_from_grid() ## funkcja zwraca prawidlowo wartości w formacie macieży z wartościami 0 i 1
        noisy_image_data = self.grid.get_noise_from_grid() ## funkcja jest do zaimplementowania powinna zwracać tablice w formacie grid z wartościami 0 i 1
        merged_image = image_data + noisy_image_data
        print (merged_image) ## return grid with numbers 0, 1
        image_data = np.array(merged_image, dtype=np.uint8) * 255
        skimage.io.imsave(filepath, image_data)
        QMessageBox.information(self, "Save", f"Grid state saved to folder {folder_index} as {filename} successfully!")


## /* funkckje aktualizują stan grida przesuwając go *\ ##

    def move_left(self):
        image = self.grid.get_image_from_grid()
        image = np.roll(image, -1, axis=1)
        image[:, 0] = 0
        self.grid.update(image)

    def move_right(self):
        image = self.grid.get_image_from_grid()
        image = np.roll(image, 1, axis=1)
        image[:, -1] = 0
        self.grid.update(image)

    def move_up(self):
        image = self.grid.get_image_from_grid()
        image = np.roll(image, -1, axis=0)
        image[-1, :] = 0
        self.grid.update(image)

    def move_down(self):
        image = self.grid.get_image_from_grid()
        image = np.roll(image, 1, axis=0)
        image[0, :] = 0
        self.grid.update(image)

    def update_cell_from_image(self, image, row, col):
        color = "black" if image[row][col] else "white"
        self.buttons[row][col].setStyleSheet(f"background-color: {color};")
    
    def show_help(self):
        msg = QMessageBox()
        msg.setWindowTitle("Help")
        msg.setText("Select Mode: Click on a cell to select it.\n"
                    "Deselect Mode: Click on a cell to deselect it.\n"
                    "Keyboard Shortcuts: Enable/Disable select/deselect modes.\n"
                    "Ctrl + S: Save the current grid state.\n"
                    "Ctrl + A: Enable Select Mode.\n"
                    "Ctrl + Q: Enable Deselect Mode.\n"
                    "Ctrl + X: Clear the grid.\n"
                    "Ctrl + H: Show this help message.\n"
                    "Click 'o' add noise\n"
                    "Click 'h' help")
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def back_to_paint_mode(self):
        self.grid = Grid(GRID_WIDTH, GRID_HEIGHT, CELL_SIZE, add_noise_function=self.add_noise)
        self.create_buttons()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.clear()  
        if event.key() == Qt.Key_X:
            self.clear()  
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_H:
            self.show_help() 
        if event.key() == Qt.Key_H:
            self.show_help()  
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_A:
            self.enable_select_mode() 
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_Q:
            self.enable_deselect_mode() 
        if event.key() == Qt.Key_B or event.key() == Qt.Key_Space or event.key() == Qt.Key_N or event.key() == Qt.Key_M:
            self.grid.selection_mode = SelectionMode.NONE
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_S:
            self.save_button.click()
        if event.key() == Qt.Key_O:
            self.add_noise_to_grid()
        if Qt.Key_0 <= event.key() <= Qt.Key_9:
            folder_index = event.key() - Qt.Key_0 
            self.save(folder_index) 
        if event.key() == Qt.Key_Backspace:
            self.back_to_paint_mode()
        if event.key() == Qt.Key_K:
            self.calculate_correlation()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Interface()
    sys.exit(app.exec_())


# Na przyszły tydzień
# Biostatystyka kolos
# Oddać program na sieci neuronowe
# Angielski  
