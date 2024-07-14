# Car-Bike Image Classification

## Project Overview
This project implements a deep learning model to classify images as either cars or bikes. It uses a pre-trained ResNet18 model, fine-tuned on a custom dataset of car and bike images.

## Features
- Utilizes transfer learning with a pre-trained ResNet18 model
- Custom dataset handling for car and bike images
- Data augmentation techniques for improved model generalization
- Training and validation pipeline
- Model saving and loading for inference
- Simple prediction script for classifying new images

## Technologies Used
- Python 3.9
- PyTorch
- torchvision
- PIL (Python Imaging Library)

## Installation

1. Clone the repository:

git clone [https://github.com/your-username/car-bike-classification.git](https://github.com/BimsaraS99/ResNet18-CarBike-Classifier-PyTorch)

2. Install the required dependencies:

pip install torch torchvision pillow


## Usage

### Training the Model

1. Prepare your dataset in the following structure:

Car_Bike_Dataset

├── Car_Bike_Dataset/
│   ├── train/
│   │   ├── Car/
│   │   └── Bike/
│   └── val/
│       ├── Car/
│       └── Bike/


Used Dataset: https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset

2. Run the training script

3. The trained model will be saved as `car-bike_classification_model.pth`.

### Making Predictions

1. Place the image you want to classify in the project directory.

2. Run the prediction script.

3. The script will output the predicted class (Car or Bike).

## Model Architecture
- Base model: ResNet18
- Modified final fully connected layer for binary classification
- Transfer learning approach with fine-tuning

## Results
The model achieves high accuracy on the validation set, demonstrating its effectiveness in distinguishing between cars and bikes in images.

## Future Improvements
- Expand the dataset to include more diverse images
- Experiment with other pre-trained models (e.g., ResNet50, EfficientNet)
- Implement real-time classification using a webcam feed

## Contributing
Contributions to improve the Car-Bike Classification project are welcome. Please feel free to submit a Pull Request.

## License
[MIT License](https://opensource.org/licenses/MIT)

## Acknowledgements
- PyTorch team for the excellent deep learning framework
- torchvision for providing pre-trained models and useful transforms
