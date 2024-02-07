# Cryptonite Research
Research on AI and multimodality for spam detection/prevention

| Model | Function | Dataset rows | Accuracy |
|-|-|-|-|
|[NLP model with unimodality](model.py)|SMS spam detection|[5574](datasets/sms_spam.csv)|0.9865471124649048|
|[NLP model with multimodality (dual)](bimodel.py)|SMS + email spam detection|[5574](datasets/sms_spam.csv) + [5754](datasets/email_spam.csv) = 11478|0.9878923892974854|
