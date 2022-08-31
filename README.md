<html>
<body>
  <h3>Using CNN models along with Merge Sort algorithm to Sort Flood Images accoding to Severity</h3>
<p align="justify"> The flood dataset was formed by scraping images from the internet. Theere are total three classes from lowest to highest severity. More details on the dataset can be found in this link https://www.kaggle.com/datasets/shawkhibnerashid/flood-dataset. You can also download the dataset from the same link. We have used DenseNet and Inception v3 models for this work. The output of the models are then sorted to sort the images. The following image shows some examples from the dataset with their corresposnding labels.</p>
<a href="https://drive.google.com/uc?export=view&id=12XGnDiWzseZWYizbx8EbgFFqXBdjAqd-"><img src="https://drive.google.com/uc?export=view&id=12XGnDiWzseZWYizbx8EbgFFqXBdjAqd-" style="width: 100%; max-width: 100%; height: 100%" title="Some examples from the dataset along with their respective labels." />
<h3>Train and Test the Model</h3>
  <p align="justify">
  First, install the required libraries from the requirements.txt file. Then, run the modelinit.py file. Make sure that the train, test and validation sets are in the same directory as the codes are. Alternatively, you can change the directory in the code. The input size has to match with the model being used. For example, the input size needs to be 224 for DenseNet and 299 for Inception v3. To test the model run the file inference.py file by specifying which model you are using and where the weight for the model is. The train procedure saves the weights after training the particular model.
  </p>
  
  
  
</body>
</html>
