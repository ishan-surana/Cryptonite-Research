# Cryptonite Research
Research on AI and multimodality for spam detection/prevention

| Model | Function | Dataset rows | Accuracy (avg.)|
|-|-|-|-|
|[NLP model with unimodality](model.py)|SMS spam detection|[5574](datasets/sms_spam.csv)|~[98.65%](recorded%20outputs/single%20modal%20output.txt)|
|[NLP model with multimodality (dual)](bimodel.py)|SMS + email spam detection|[5574](datasets/sms_spam.csv) + [5754](datasets/email_spam.csv) = 11478|~[98.79%](recorded%20outputs/bimodal%20output.txt)|