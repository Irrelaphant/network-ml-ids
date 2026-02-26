First Checkpoint:

    1. Create a full pipeline that can take the training dataset, process it, then train a model.
    2. Baseline model and it's saved artifacts are actually able to "scan" and produce output.
    3. CSV with scored output includes alerts and the alerts can be tuned by adjusting the threshehold.

Second Checkpoint:

    1. Polish scan output, have CSV sort most malicious to least
    2. Severity labels, and more diagnostic info included such as scored rows, alert count, etc.
    3. Demo an actual CLI wrapper that makes this program much easier to use.
    4. CLI should be pointed to where dataset is stored, easily usable and can build, train and scan with a single command. Users shouldn't have to reconfigure everytime they want to use.
    5. README should be a quicksart guide and work on a VM or lightweight system like a Raspberry Pi

Third Checkpoint (Final Deliverable):
    
    1. A single unified interface, that'll work like "python /scripts/mlids.py --input data.csv --model rf_full.joblib --output date.csv" 
    2. Train using datasets that have other information more usable for a SOC, perhaps simulate attacks in VM environment, then train model off that.
    3. Needs to be able to be trained off of more than just the CIC-2017 dataset, should make something that can take apart pfSense firewalls and other common firewall logs.
    4. Final write up on everything