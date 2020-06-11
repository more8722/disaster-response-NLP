# Disaster-response-Pipeline NLP
This project consists of a disaster response web-app. Include ELT pipeline, NLP processing and machine learning pipeline. 
# Installations 
 The following libraries meed to be installed  
NumPy
Pandas
Matplotlib
Json
Plotly
Nltk
Flask
Sklearn
Sqlalchemy
Sys
Re
Pickle
# Motivation
 Using text data from figure-8, to bulid ETL pipeline. then to train a machine learning model to classify disaster messages. Finally, by using web app that worker can input messages then get classification to serveral categories , such as "water", "food", etc.  
  
### For data       
1.Messages data: disaster_messages.csv      
2.Categories data: disaster_categories.csv       
3.SQL Database: DisasterResponse.db      
4.Jupyter notebook for building ETL pipeline: ETL Pipeline Preparation.ipynb     
5.Python script for processing the data: process_data.py         


### Models
1.To a machine learning pipeline: ML Pipeline Preparation.ipynb    
2.get the training file: train_classifier.py   
3. A pickle file that contains the trained model: classifier.pkl    

### Web app                                     
1.Run run.py for web app            
2.Enter messages you want                               

### Run program                          
1.python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db                         
2.python train_classifier.py DisasterResponse.db classifier.pkl                               
3.python run.py                          
4.Go to http://0.0.0.0:3001/ 

### Performance test       

![image](https://github.com/more8722/disaster-response-NLP/blob/master/message%20distribution.PNG)           
![image](https://github.com/more8722/disaster-response-NLP/blob/master/message.PNG)
