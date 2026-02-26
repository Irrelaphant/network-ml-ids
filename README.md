First Checkpoint, 2/25/2026

This project is a machine learning based intrusion detection system that looks at network traffic flow and creates an evilness prediction using a machine learning model. The end deliverable goal is to have a CLI-Style interface that allows a user to build a dataset from their own CSV files or other public datasets. Then they can easily train and tune their own machine learning model, with easy adjustment knobs to tune the false-positive noise. Because this project uses more lightweight flow metadata and features of network traffic to produce alerts (instead of raw packets), it's possible to run the IDS on a platform like a Raspberry Pi.  

The CIC-2017 dataset in this project is a flow feature table, meaning each row represents a network flow (a summarized connection), and each column represents a statistic such as flow durection, packet rate, inter-arrival time, TCP flag counts, and destination ports. The dataset also importantly labels traffic as benign or malicious, allowing us to train the model on what malicious network flow looks like (in the dataset). 

Building the dataset:
    01_build_dataset.py's purpose is to convert all the CSV's in the dataset into one machine learning ready dataset, it strips down column whitespace, 
		normalizes naming, converts invalid values to NaN, adds a source_file metadata segment to allow for day specific training, and lastly it outputs the combined CSV to data/processed. 
		The training here requires very consistent features and labeling for the best performance and accuracy, which is why I can't just download other datasets and use them (yet!!), I'd have to "clean them up first" to make them model ready. I believe the next step of this project is intrudocuing more data for the model to train off of, because in hindsight, a weeks worth of internet traffic is not enough to train an actual IDS. 

Training the model:
    02_train.py's purpose is to train and evaluate a baseline Machine Learning classifier, and save artifacts. 
		It loads the dataset made by 01_build_dataset.py, and is set up to train on the Monday through Thursday, but not Friday. 
		A Random Forest Classifier is then made using the scikit-learn library, and the trained model is output as rf_(name).joblib, the features output as features_(name).json, and the confusion matrix and precision is output in metrics_(name).json. 
		The training features are important to save as the same columns and order from the input CSV has to match if you wanted to train the model more using other CSV's or for future testing and use of the model. For example, if training on Mon-Thurs, and testing on Friday, Friday needs it's features to match. This is incredibly limiting and I will work to train this. I honestly think the answer is just more datasets to train the model off of to gain more features. 

Scanning and using the model:
    03_predict.py's purpose is to use the trained model to create a score CSV and IDS style alerts. 
		It loads in the model and the model's features and processes the inputted CSV traffic into numeric features. 
		Then it aligns the feature scheme with the training scheme, and scores each row with a prediction. 
		The threshehold of the model dictates whether or not there is an alert. 
		This outputs a probability of maliciousness score, a predicted label, and any other metadata fields that were included in the input.
		This is still very limited, because the CIC dataset does not include source IPs and destination IPs, in a real SOC environment, pfSense firewall logs would provide IP context. 
		For the next checkpoint, I will try and get this program to be compatible with pfSense firewalls, and instead train the model on my virtual environment. 
		Currently, the results mostly show very high precision on malicious traffic, but a ton of missed alerts.
		This dataset and configuration (in it's current tuning) is very conservative. 
    	This conservative model can be adjusted with threshehold tuning, which changes the amount of alert volume produced. For example, if trained on Mon-Thurs, then tested on Fri, threshehold 0.9 reveals only 30 alerts, threshld 0.7 returns about 40, and threshehold 0.3 returns around 217. Essentially, this is a very finnicky false-positive knob essentially. 



    
