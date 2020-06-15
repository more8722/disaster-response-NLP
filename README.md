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
## Motivation
 Using text data from figure-8, to bulid ETL pipeline. then to train a machine learning model to classify disaster messages. Finally, by using web app that worker can input messages then get classification to serveral categories , such as "water", "food", etc.  
  
### ETL Pipeline       
1.Load Messages data: messages.csv      
2.Load Categories data: categories.csv   
3.Merge and clean data
3.Saved to SQL Database: DisasterResponse.db      
4.Jupyter notebook for building ETL pipeline: ETL Pipeline Preparation.ipynb     
5.Python script for processing the data: process_data.py         


### ML Pipeline
1.Load SQLite Datbase:DisasterResponse.db        
2.Splits the dataset into training and test sets               
3.Builds a text processing and machine learning pipeline               
4.Trains and tunes a model using GridSearchCV                     
5.Outputs results on the test set                          
6.Exports the final model as a pickle file                               
7.design Machine learning pipeline: train_classifier.py                                               
8.Exports pickle file that contains the trained model: classifier.pkl                                                 

### Web app                                     
1.Run run.py for web app            
2.Enter messages you want                               

### Project design                  
- app              
| - template                       
| |- master.html # main page of web app                        
| |- go.html  # classification result page of web app                     
|- run.py  # Flask file that runs app                          

- data                               
|- categories.csv  # data to process                    
|- messages.csv  # data to process                                
|- process_data.py                                
|- DisasterResponse.db   # database to save clean data                                    

- models                      
|- train_classifier.py                
|- classifier.pkl  # saved model                           
                  
- README.md                             

### Run program                          
1.python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db                         
2.python train_classifier.py DisasterResponse.db classifier.pkl                               
3.python run.py                          
4.Go to http://0.0.0.0:3001/ 

### Performance test       

![image]https://github.com/more8722/disaster-response-NLP/blob/master/message%20distribution.PNG)       

