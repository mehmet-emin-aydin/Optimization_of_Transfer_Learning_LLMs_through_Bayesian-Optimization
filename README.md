# Turkish Language Models on Hugging Face

This repository contains the code and models for training and deploying multiple Turkish language models on Hugging Face. The project utilizes datasets from Turkish Wikipedia and sports data. The goal is to create, train, and evaluate these models, saving intermediate and final versions to the Hugging Face Hub for public use.

## Project Structure

### 1. Data Cleaning and Preprocessing
- **Script**: `1.1_DataCleaning.py`
- **Description**: This script cleans and preprocesses the Turkish Wikipedia data by removing unwanted content, splitting into sentences, and fixing special cases.

### 2. Tokenization
- **Script**: `1.2_Tokenizer.py`
- **Description**: This script creates and saves a tokenizer trained on the cleaned dataset.

### 3. Model Training
- **Script**: `2_Model_M0.py`
  - **Description**: Trains the M0 model on the cleaned Wikipedia data.
- **Script**: `3.1_Model_M1.py`
  - **Description**: Trains the M1 model on sports dataset by fine-tuning M0 model.
- **Script**: `3.2_Model_M2.py`
  - **Description**: Trains the M2 model on policy dataset by fine-tuning M0 model.

### 4. Advanced Model Training
- **Script**: `5.1_Model-A-B-C.py`
  - **Description**: Trains models A, B, and C on various datasets by fine-tuning pretrained models.
- **Script**: `5.2_Model_D.py`
  - **Description**: Trains model D on both sports and policy dataset by fine-tuning M0 model.

### 5. Evaluation
- **Script**: `6_Success_Score_Calculation.py`
  - **Description**: Calculates and plots the success score of the trained models by evaluating them on specific sub-datasets.

## Training Steps

1. **Data Cleaning**: Run `1.1_VeriTemizleme.py` to clean and preprocess the dataset.
2. **Tokenization**: Run `1.2_Tokenizer.py` to create and save the tokenizer.
3. **Model Training**:
   - Run `2_M0_Modeli.py` to train the M0 model.
   - Run `3.1_M1_Modeli.py` to train the M1 model.
   - Run `3.2_M2_Modeli.py` to train the M2 model.
4. **Advanced Model Training**:
   - Run `5.1_Model-A-B-C.py` to train models A, B, and C.
   - Run `5.2_Model_D.py` to train model D.
5. **Evaluation**: Run `6_Basari_Skoru_Hesaplama.py` to evaluate the models and plot the results.


## Model Configurations

- **M0 Model**: Custom configuration for GPT-2, trained on cleaned Wikipedia data.
- **M1 Model**: Trained on sports data, utilizing an existing trained model for further training.
- **M2 Model**: Trained on policy data, utilizing an existing trained model for further training.

### Training Parameters
- Learning rate: 5e-5
- Batch size: Auto-detected
- Number of epochs: 10
- Early stopping: Enabled with a patience of 3 epochs
- Gradient accumulation steps: 8
- Weight decay: 0.01
- FP16: Enabled
- Save strategy: Epoch-based, limited to 2 best models

## Datasets

1. [Turkish Wikipedia Data](https://drive.google.com/file/d/1-6Xdvn_R7LbPGpU3wzP2B-ZtYq54WV4K/view?usp=sharing) (Used for M0 Model)
2. [Sports Data](https://drive.google.com/file/d/1lYxotjuIdYzOvm2eiMki8lj588v7guPJ/view?usp=sharing) (Used for M1 Model)
3. [Politics Data](https://drive.google.com/file/d/1lYxotjuIdYzOvm2eiMki8lj588v7guPJ/view?usp=sharing) (Used for M2 Model)

## Running the Project

To run the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/turkish-language-models.git
    cd turkish-language-models
    ```
3. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```
5. Data Cleaning:

   Run 1.1_VeriTemizleme.py to clean and preprocess the dataset:
   ```sh
   python 1.1_VeriTemizleme.py
   ```
7. Tokenization:
   Run 1.2_Tokenizer.py to create and save the tokenizer:
   ```sh
   python 1.2_Tokenizer.py
   ```
8. Model Training:
   Run 2_M0_Modeli.py to train the M0 model:
   ```sh
   python 2_M0_Modeli.py
   ```
   Run 3.1_M1_Modeli.py to train the M1 model:
   ```sh
   python 3.1_M1_Modeli.py
   ```
   Run 3.2_M2_Modeli.py to train the M2 model:
   ```sh
   python 3.2_M2_Modeli.py
   ```
7. Advanced Model Training:
   Run 5.1_Model-A-B-C.py to train models A, B, and C:
   ```sh
   python 5.1_Model-A-B-C.py
   ```
   Run 5.2_Model_D.py to train model D:
   ```sh
   python 5.2_Model_D.py
   ```
7. Evaluation:
   Run 6_Basari_Skoru_Hesaplama.py to evaluate the models and plot the results:
   ```sh
   python 6_Basari_Skoru_Hesaplama.py
   ```





## Contact

For any questions or issues, please contact me at 16mehmet.emin@gmail.com

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
