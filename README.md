# Speech To Text

### Abstract
This report details a speech-to-text system using Convolutional Neural Networks (CNNs) and the TIMIT dataset, focusing on transcribing speech into text. The proposed model outperforms a baseline model in speech recognition performance.

### Introduction
The project develops a speech-to-text system using CNNs trained on the TIMIT dataset. The architecture includes convolutional, pooling, and dense layers, aiming to improve upon traditional speech recognition systems.

### Dataset
- **Data Source**: TIMIT dataset, featuring phonemically and lexically transcribed recordings of American English speakers.
- **Preprocessing**: Includes loading audio data, converting to spectrograms, padding for uniformity, and preparing for CNN processing.

### Methods
- **Baseline Model**: Simple architecture consisting of Flatten, Dense, Reshape, and TimeDistributed layers.
- **CNN Model**: Advanced architecture with multiple convolutional, pooling, and dense layers, designed for capturing complex patterns in spectrograms.

### Results & Analysis
- **Baseline Model Performance**: Demonstrated low training and validation accuracy.
- **CNN Model Performance**: Significantly higher training and validation accuracy, indicating effective learning of patterns in input spectrograms.
- **Improvement Over Baseline**: CNN model showed a substantial increase in accuracy, demonstrating the benefits of convolutional architecture in speech recognition tasks.

### Discussion and Summary
- **Findings**: CNN model's architecture proved more effective than the baseline model in speech recognition.
- **Future Work**: Suggestions for hyperparameter tuning, employing advanced techniques like batch normalization, and exploring alternative architectures for further improvements.



