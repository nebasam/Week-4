# African language Speech Recognition - Speech-to-Text
The World Food Program wants to deploy an intelligent form that collects nutritional information of food bought and sold at markets in two different countries in Africa - Ethiopia and Kenya. The design of this intelligent form requires selected people to install an app on their mobile phone, and whenever they buy food, they use their voice to activate the app to register the list of items they just bought in their own language. The intelligent systems in the app are expected to live to transcribe the speech-to-text and organize the information in an easy-to-process way in a database.

## Folder structure for branch
* notebook-contains notebooks for describing the functionality of the the classes to achieve the meta generation and the preprocessing
* modules-contains scripts for Meta generation, preprocessing and feature extraction
* data.dvc- DVC File for versioning of the data
* requirements.txt- dependencies for code inside this branch

## Data
* Dataset for Swahili- https://github.com/getalp/ALFFA_PUBLIC

## Data Features
Input features (X): audio clips of spoken words
Target labels (y):  text transcript of what was spoken