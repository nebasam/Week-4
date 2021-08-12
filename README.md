#  Speech Recognition
## Live Transcription of Swahili Audio to Swahili Text

### Navigation
- [Introduction](#introduction)
- [Objective](#objective)
- [Data](#data)
- [Notebooks](#notebooks)
- [Scripts](#scripts)
- [Testing](#testing)
- [Model](#model)
- [Deployment](#deployment)
- [Contributors](#contributors)

### Introduction
World food Program wants to collect nutritional information of food bought and sold in Kenya. The project is designed to have selected people install an app on their mobile phones, and whenever they buy food, they use their voices to activate the app to register the list of items they have bought in Swahili. The app is expected to live transcribe the voice of the people to text and organize the information in an easy-to-process way in a database
### Objective
This project builds, trains and deploy a deep learning model which transcribe audio in Swahili to text in Swahili.
### Data
<ul>
<li>Dataset for Swahili-  https://github.com/getalp/ALFFA_PUBLIC</li></ul>
<br/>

### Data Features
    Input features (X): audio clips of spoken words
    Target labels (y):  text transcript of what was spoken

## Directory Structure 

<ul>
    <li><b>artifacts</b>-contains artifacts such meta files and other artifacts generated through the project</li>
    <li>###notebook-contains notebooks for describing the functionality of the the classes to achieve the meta generation and the preprocessing</li>
    <li>###scripts-contains scripts for Meta generation, preprocessing and feature extraction</li>
    <li>###data.dvc- DVC File for versioning of the data</li>
    <li>###requirements.txt- dependencies for code inside this branch</li>
</ul>

### Testing
### Modelling
### Deployment
## Contributors
1. [Michael Darko Ahwireng](https://github.com/mdahwireng)



