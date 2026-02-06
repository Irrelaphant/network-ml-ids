Dataset Build script
Baseline Training script
Prediction script
Evaluation report saved to /reports with a confusion matrix 
README explains how to run on a new machine/VM

Capstone goals
    Day based split evaluations, train on mon-thurs, test on friday
    error analysis on false positives and false negative examples
    feature importance plot/report
    threshold tuning and rationale behind tuning
    "dmeo modes" that will make it feel like a useful tool

TO-DO for later

figure out how to train the model on mon-thurs, and then test it on friday traffic
create a demo pythons script



To create changes for github repo:
git add .
git commit -m "Describe change"
git push

to run python demo: 
python scripts/04_demo.py --input "data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv" --threshold 0.8 --rebuild --retrain
