#  Speech Recognition
## Live Transcription of Swahili Audio to Swahili Text

### Navigation
- [Introduction](#introduction)
- [Objective](#objective)
- [Data](#data)
- [Data Features](#data_features)
- [Directory Structure](#directory_structure)
- [Testing](#testing)
- [Model](#model)
- [Deployment](#deployment)
- [Contributors](#contributors)

### Introduction
<p>World food Program wants to collect nutritional information of food bought and sold in Kenya. The project is designed to have selected people install an app on their mobile phones, and whenever they buy food, they use their voices to activate the app to register the list of items they have bought in Swahili. The app is expected to live transcribe the voice of the people to text and organize the information in an easy-to-process way in a database</p>

### Objective
This project builds, trains and deploy a deep learning model which transcribe audio in Swahili to text in Swahili.

### Data

<ul>
<li>Dataset for Swahili-  https://github.com/getalp/ALFFA_PUBLIC</li>
</ul>

### Data_Features
    Input features (X): audio clips of spoken words
    Target labels (y):  text transcript of what was spoken

## Directory_Structure 

<ul>
    <li><b>Artifacts</b>-A directory which contains artifacts such meta files and other artifacts generated through the project</li>
    <li><b>Notebook</b>-A directory which contains notebooks for describing the functionality of the the classes to achieve the meta generation and the preprocessing</li>
    <li><b>Scripts</b>-A directory which contains scripts for Meta generation, preprocessing and feature extraction </li>
    <li><b>data.dvc</b>- DVC File for versioning of the data</li>
    <li><b>requirements.txt</b>- A file for dependencies for the project</li>
</ul>

### Testing
### Modelling
### Deployment
## Contributors
1. [Michael Darko Ahwireng](https://github.com/mdahwireng)
2. [Toyin Hawau Olamide](https://github.com/theehawau)



