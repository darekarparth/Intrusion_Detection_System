# Intrusion_Detection_System
I have designed an Intrusion Detection System using Machine Learning, Deep Learning, based on CICIDS2017 Dataset.
I’m thrilled to share the project I recently completed as part of my Semester-1 Cybersecurity coursework at Ulster University — building an Intrusion Detection System (IDS) using Machine Learning and Deep Learning on the CIC-IDS2017 dataset!

Objective:
To design a robust IDS that can accurately detect a wide range of cyberattacks, including DDoS, SQL Injection, Port Scans, and more, using real-world enterprise-level network traffic data.

Dataset Used:
[CIC-IDS2017] by the Canadian Institute for Cybersecurity — 2.8 M+ labeled flow records with 79 features and 14 attack classes.

Models Developed:
Random Forest
XGBoost
Deep Learning (3-layer dense network)

Key Steps:
Cleaned and preprocessed millions of rows using Pandas & Scikit-learn
Tackled class imbalance using SMOTE.
Evaluated model performance with confusion matrices, F1-scores, and accuracy.
Integrated real-time intrusion detection with CICFlowMeter and Python's Watchdog for monitoring live logs.

Results:
Random Forest + SMOTE achieved ~99% accuracy with balanced detection across all classes
XGBoost + SMOTE performed exceptionally on rare attacks like Heartbleed & Infiltration
Deep Learning improved with SMOTE but lagged behind ensemble models

Live Detection Capability:
The system scans incoming .csv log files in real time and raises alerts for attacks — a small yet practical simulation of real-world SOC environments.

Why It Matters:
This project reflects how AI-powered IDS can supplement traditional defenses and detect sophisticated threats more effectively, even in imbalanced and noisy environments.
