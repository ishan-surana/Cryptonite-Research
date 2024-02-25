# Cryptonite Research
Research on AI and multimodality for spam detection/prevention
<br><br>
Initial Work
| Model | Function | Dataset rows | Accuracy (avg.)|
|-|-|-|-|
|[NLP model with unimodality](Initial%20work/model.py)|SMS spam detection|[5574](Initial%20work/datasets/sms_spam.csv)|~[98.65%](Initial%20work/recorded%20outputs/single%20modal%20output.txt)|
|[NLP model with multimodality (dual)](Initial%20work/bimodel.py)|SMS + email spam detection|[5574](Initial%20work/datasets/sms_spam.csv) + [5754](Initial%20work/datasets/email_spam.csv) = 11478|~[98.79%](Initial%20work/recorded%20outputs/bimodal%20output.txt)|

<br>

Model implementation [main paper](https://www.mdpi.com/2504-4990/5/3/58#B27-make-05-00058)
| Model | Function | Dataset rows | Accuracy (avg.)|
|-|-|-|-|
|[Behzadan model recreation (Reference 28)](Model%20Recreation/Behzadan%20(Citation%2028)/model.py)|Only text is analysed|[21368]()|~[96.79%](Model%20Recreation/Behzadan%20(Citation%2028)/output.txt)|
|[Actual recreation](Model%20Recreation/Behzadan%20(Citation%2028)/complex_model.py)|Text + tweet details analysed|[21368]|~[-](#)|
