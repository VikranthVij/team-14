#!/bin/bash

# Download PlantVillage dataset for pest detection
wget https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded -O plantvillage_dataset.zip
unzip plantvillage_dataset.zip -d pest_images

# Download soil parameter datasets
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00697/Soil_Moisture_Dataset_2015-2016.zip
unzip Soil_Moisture_Dataset_2015-2016.zip -d soil_data

# Download audio pest dataset
mkdir -p audio_data
# Note: This is a placeholder - actual audio dataset needs to be collected
# Create empty directories for audio data collection
mkdir -p audio_data/pest_activity audio_data/environmental

# Download pre-trained models
mkdir -p models
wget https://storage.googleapis.com/aihub-public-models/tflite_models/pest_detection.tflite -O models/pest_detection.tflite
wget https://storage.googleapis.com/aihub-public-models/tflite_models/soil_analysis.tflite -O models/soil_analysis.tflite
wget https://storage.googleapis.com/aihub-public-models/tflite_models/audio_classification.tflite -O models/audio_classification.tflite

# Clean up
rm *.zip
