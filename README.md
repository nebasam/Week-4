#  Speech Recognition
## Live Transcription of Swahili Audio to Swahili Text

### Navigation
- [Speech Recognition](#speech-recognition)
  - [Live Transcription of Swahili Audio to Swahili Text](#live-transcription-of-swahili-audio-to-swahili-text)
    - [Navigation](#navigation)
    - [Introduction](#introduction)
    - [Objective](#objective)
    - [How to start](#how-to-start)
    - [Data](#data)
    - [Data_Features](#data_features)
  - [Directory_Structure](#directory_structure)
    - [Testing](#testing)
    - [Modelling](#modelling)
    - [Deployment](#deployment)
  - [Contributors](#contributors)

### Introduction
<p>World food Program wants to collect nutritional information of food bought and sold in Kenya. The project is designed to have selected people install an app on their mobile phones, and whenever they buy food, they use their voices to activate the app to register the list of items they have bought in Swahili. The app is expected to live transcribe the voice of the people to text and organize the information in an easy-to-process way in a database</p>

### Objective
This project builds, trains and deploy a deep learning model which transcribe audio in Swahili to text in Swahili.

### How to start
 * <b>Machine Setup:</b>

First, you need to have <b>python 3</b> installed.

Next clone this github link

git clone https://github.com/10Academy-Group-4/Week-4
Finally, you can install the requirements. If you are an Anaconda user: (else replace pip with pip3 and python with python3)
pip install -r requirements.txt

 * <b>Docker:</b>

This is a containerized flask application with docker image put on docker hub.A docker image is available with all pre-requisites installed. Here is how you use it

<b>Pull docker image</b>

  docker pull nebasam/stt-swahili

<b>Run docker image</b>

  docker run --rm -it  -p 33507:33507/tcp nebasam/stt-swahili:latest

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
    <li><b>test_data</b>-A directory which has data for running tests for every commit or merge on the main branch</li>
    <li><b>tests</b>-A directory which has the codes for testing  every commit or merge on the main branch</li>
    <li><b>data.dvc</b>- DVC File for versioning of the data</li>
    <li><b>requirements.txt</b>- A file for dependencies for the project</li>
</ul>

### Testing
<p> The inbuit <b>unittest</b> library in python was used to for the testing of the functions and classes in the project. A <b>.travis.ymal</b> was added to automate testing of any commit or merge made to the main branch. Data used for testing is found in test_data directory</p>

### Modelling
To get an idea of how models are setup and investigated, take a look at the notebooks for Models, WordError and Augmentation.



### Deployment
<p>The user interface was built with <b>flask</b>. The model was dockerized and deployed on <b>Heroku on https://swahili-stt.herokuapp.com/</p>

## Contributors
1. [Michael Darko Ahwireng](https://github.com/mdahwireng)
2. [Toyin Hawau Olamide](https://github.com/theehawau)
3. [Nebiyu Samuel](https://github.com/nebasam)
4. [Sibitenda Harriet](https://github.com/SibitendaHarriet)
5. [Same Michael](https://github.com/SameC137)
6. [Mubarak Sani](https://github.com/SamDewriter)
7. [Khairat Ayinde](https://github.com/khaiyra)
