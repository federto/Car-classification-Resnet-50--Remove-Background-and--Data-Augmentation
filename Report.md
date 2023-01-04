The Proyect has the objective of identify various cars models 196 exactly,  for that task we count with a small image dataset and the correspondent labels in a csv file. 
We can separate the entire project in 3 steps :

- Data Managemt
- Train
- Evaluation


Data Managemt

First, we connect to our aws server and create a personal folder, where all the project is saved. So when we have our carpet we proced to download the image dataset
and the csv file with the labels of the cars models from our aws server. Later we separete the dataset in train / test carpets with the information from the csv.

Train

For the train, we must complete various functions that habilite us to do all the funcionalities like data augmentation, predict, model and utils.

    - Data Augmentation :  is a process of artificially increasing the amount of data by generating new data points from existing data, with data agumentations parameters
                           we mirror the images, rotate a few grades and make a little of zoom to get new data points.
    
    - Model :  set the model parameters

    - Predict : get the predictions label and probability using our model from the test carpet

    - Utils :  loas the yml file with the model parameters

Evaluation 

After experimenting with many parameters of the model in the yml files, we found that rotating the photo a lot would leave the car in a vertical position, which is of no use to us. A low learning rate, a high dropout, and a small zoom would be the ideal parameters.


After all, we decided to crop the cars in the photos to train the model again with the best yml file, and see if we could get a better result. 

And we get results! - the model gained 0.23 points, reaching an accuracy = 0.7688  

Here we got the model performance in train https://tensorboard.dev/experiment/uPN8jEUORfqRaVndGr0jXw/#scalars


