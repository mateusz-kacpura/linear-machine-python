import numpy as np
import skimage.io
import skimage.transform
import os
import random
from skimage.color import rgb2gray
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QGridLayout, QVBoxLayout, QMessageBox, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt
import sys

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

## Grid (px)
GRID_WIDTH = 20
GRID_HEIGHT = 20
CELL_SIZE = 25
IMAGE_SIZE = 20

# Noise <0,1>
NOISE_LEVEL_MIN = 0
NOISE_LEVEL_MAX = 0.1

## Linear perceptron
NUM_CATEGORIES = 10
ITERATIONS = 30
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

class CNN:
    print ("Algorytm CNN nie został jeszcze zaimpolementowany")
    pass

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

    def predict(self, x):
        return self.output(x)

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
        self.save_button = QPushButton("Save (Ctr + s)")
        self.select_button = QPushButton("Select Mode (Ctr + a)")
        self.deselect_button = QPushButton("Deselect Mode (Ctr + q)")
        self.train_linear_button = QPushButton("Liner Machine training")
        self.train_cnn = QPushButton("CNN training")
        self.test_button = QPushButton("Test")
        self.clear_button = QPushButton("Clear (Ctr + x)")
        self.help_button = QPushButton("Help (Ctr + h)")
        self.move_left_button = QPushButton("← Move Left")
        self.move_right_button = QPushButton("→ Move Right")
        self.move_up_button = QPushButton("↑ Move Up")
        self.move_down_button = QPushButton("↓ Move Down")
        
        arrow_button_layout = QHBoxLayout()
        arrow_button_layout.addWidget(self.move_left_button)
        arrow_button_layout.addWidget(self.move_up_button)
        arrow_button_layout.addWidget(self.move_right_button)
        arrow_button_layout.addWidget(self.move_down_button)

        other_button_layout = QVBoxLayout()
        other_button_layout.addWidget(self.save_button)
        other_button_layout.addWidget(self.select_button)
        other_button_layout.addWidget(self.deselect_button)
        other_button_layout.addWidget(self.train_linear_button)
        other_button_layout.addWidget(self.train_cnn)
        other_button_layout.addWidget(self.test_button)
        other_button_layout.addWidget(self.clear_button)
        other_button_layout.addWidget(self.help_button)

        button_layout = QVBoxLayout()
        button_layout.addLayout(arrow_button_layout)
        button_layout.addLayout(other_button_layout)

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
        self.train_cnn.clicked.connect(self.train_CNN)
        self.test_button.clicked.connect(self.test)
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

    def preprocess_training_data(self):

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
                        
                        global NUMBERS_PHOTO_MULTIPLER
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
        
        return np.array(training_data), np.array(number_machine)

    def train_linear_machine(self):

        training_data, number_machine = self.preprocess_training_data()

        N = GRID_WIDTH

        # Inicjalizacji maszyn liniowych
        self.linearmachines = []
        for i in range(N):
            self.linearmachines.append([])
            for j in range(N):
                self.linearmachines[i].append(LinearMachine(N * N, NUM_CATEGORIES))

        # Trening maszyny liniowej
        total_correct = 0
        total_samples = len(number_machine)
        for i in range(N):
            for j in range(N):
                y = [number_machine[index] for index in range(len(training_data))]
                self.linearmachines[i][j].train(training_data, y)

                correct = 0
                for index, x in enumerate(training_data):
                    prediction = self.linearmachines[i][j].predict(x)
                    if prediction == number_machine[index]:
                        correct += 1
                total_correct += correct

        # Dokładność
        overall_accuracy = total_correct / (N * N * total_samples)

        print(f"Overall accuracy: {overall_accuracy}")
        print("Done!")

    def train_CNN(self):
        print ("Traing CNN nie może zostać zainicjowany z powodu braku implementacji classy")

    def test(self):
        if not hasattr(self, 'linearmachines') or self.linearmachines is None:
            print ("First you need put training")
            return

        print("Testing...")

        recognition_table = {}
        selected_cells = []

        for i in range(len(self.grid.grid)):
            for j in range(len(self.grid.grid[0])):
                if self.grid.grid[i][j]:
                    selected_cells.append((i, j))

        if not selected_cells:
            print("No cells selected.")
            return

        max_i = max(selected_cells, key=lambda x: x[0])[0] + 1
        max_j = max(selected_cells, key=lambda x: x[1])[1] + 1
        test_image = np.zeros((max_i, max_j))
        for cell in selected_cells:
            test_image[cell[0]][cell[1]] = 1

        extended_test_image = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        extended_test_image[:max_i, :max_j] = test_image

        noisy_test_image = self.add_noise(extended_test_image, NOISE_LEVEL_MAX) 
        denoised_test_image = self.denoise(noisy_test_image)  

        flat_image = denoised_test_image.reshape(-1)

        for i, j in selected_cells:
            prediction = self.linearmachines[i][j].predict(flat_image)

            if prediction in recognition_table:
                recognition_table[prediction] += 1
            else:
                recognition_table[prediction] = 1

        print("\nRecognition table:")
        for number, count in recognition_table.items():
            print(f"Number {number} recognized {count} times.")

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
