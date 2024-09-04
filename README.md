# Next-Word Predictor

This repository contains the implementation of a Next-Word Predictor using a Long Short-Term Memory (LSTM) neural network. The model is trained to predict the next word in a sequence of text, learning from user-specific data to tailor predictions based on individual writing patterns.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Prediction](#prediction)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Next-Word Predictor is designed to enhance text entry by suggesting the next word based on the context of the current sentence. The model is built using an LSTM network, which is well-suited for sequence prediction tasks due to its ability to maintain context over long sequences.

Key Features:
- Trains on user-specific data for personalized predictions.
- Supports custom text corpora for domain-specific predictions.
- Provides real-time next-word suggestions during text entry.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Harshwardhanpjadhav/Next-Word-Predictor.git
   cd Next-Word-Predictor
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

1. **Collect User Data:** Gather text data from the user, such as emails, chat logs, or notes. Store this data in a text file or any format that can be easily processed.

2. **Preprocess Data:** Tokenize the text into sequences of words and prepare it for training. Ensure that the text is cleaned and normalized.

### Training the Model

To train the LSTM model, run the following command:

```bash
python train.py --data_path data/user_data.txt --output_path models/next_word_model.h5
```

- `data_path`: Path to the text file containing the training data.
- `output_path`: Path where the trained model will be saved.

### Prediction

Once the model is trained, you can use it to predict the next word in a given sequence. Run the following command:

```bash
python predict.py --model_path models/next_word_model.h5 --input_text "I am going to"
```

- `model_path`: Path to the trained LSTM model.
- `input_text`: The text sequence for which you want to predict the next word.

### Real-Time Prediction

For real-time next-word suggestions, you can integrate the prediction script into a text editor or any text input interface. The prediction script can be modified to run continuously and suggest words as the user types.

## Model Training

The LSTM model is trained on sequences of words generated from the user's text data. The architecture includes:
- **Embedding Layer:** Converts words into dense vectors of fixed size.
- **LSTM Layers:** Captures the sequence information and context of the text.
- **Dense Output Layer:** Predicts the next word in the sequence.

### Hyperparameters

- **Sequence Length:** The number of previous words used to predict the next word.
- **Epochs:** Number of training epochs.
- **Batch Size:** Number of samples per gradient update.

These parameters can be configured in the `train.py` script.

## Customization

You can customize the model and training process by modifying the following:
- **Model Architecture:** Adjust the number of LSTM layers, units, and dropout rates in the `train.py` script.
- **Training Data:** Use a different corpus or combine multiple datasets for training.
- **Prediction Logic:** Modify the `predict.py` script to change how the next word is selected.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to suggest improvements or report bugs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

