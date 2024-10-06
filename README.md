# Cryptonite Research
Research on AI and multimodality for spam detection/prevention
<br><br>
Initial Work
| Model | Function | Dataset rows | Accuracy (avg.)|
|-|-|-|-|
|[NLP model with unimodality](Initial%20work%20%5BLampu%20(Citation%2027)%5D/model.py)|SMS spam detection|[5574](Initial%20work%20%5BLampu%20(Citation%2027)%5D/datasets/sms_spam.csv)|~[98.65%](Initial%20work%20%5BLampu%20(Citation%2027)%5D/recorded%20outputs/single%20modal%20output.txt)|
|[NLP model with multimodality (dual)](Initial%20work%20%5BLampu%20(Citation%2027)%5D/bimodel.py)|SMS + email spam detection|[5574](Initial%20work%20%5BLampu%20(Citation%2027)%5D/datasets/sms_spam.csv) + [5754](Initial%20work%20%5BLampu%20(Citation%2027)%5D/datasets/email_spam.csv) = 11478|~[98.79%](Initial%20work%20%5BLampu%20(Citation%2027)%5D/recorded%20outputs/bimodal%20output.txt)|

<br>

Model implementation [main paper](https://www.mdpi.com/2504-4990/5/3/58#B27-make-05-00058)<br>
The 3rd and 4th model have the architecture and layers as per the main paper, but operate on the dataset of reference 28 (dataset provided), and therefore, the labels are different.
| Model | Function | Dataset rows | Accuracy (avg.)|
|-|-|-|-|
|[Behzadan model basic recreation (Reference 28)](Model_Recreation/Behzadan%20(Citation%2028)/basic_model.py)|Only text is analysed|[21368](Model_Recreation/Behzadan%20(Citation%2028)/tweets.csv)|~[96.79%](Model_Recreation/Behzadan%20(Citation%2028)/basic_output.txt)|
|[Complex recreation](Model_Recreation/Behzadan%20(Citation%2028)/complex_model.py)|Text + tweet details analysed|[21368](Model_Recreation/Behzadan%20(Citation%2028)/tweets_final.csv)|~[96.70%](Model_Recreation/Behzadan%20(Citation%2028)/complex_output.txt)|
|[Recreation with exact parameters of main paper](Model_Recreation/model_with_same_parameters.py)|Text + tweet details analysed + Layers applied with exact parameters|[21368](Model_Recreation/tweets_final.csv)|~[14.67%](Model_Recreation/output_for_same_parameters.txt)|
|[Recreation with changed parameters (better output)](Model_Recreation/model.py)|Text + tweet details analysed + Layers applied with modified parameters|[21368](Model_Recreation/tweets_final.csv)|~[95.77%](Model_Recreation/output.txt)|


[Confusion matrix](Model_Recreation/confusion_matrix.png)

[Knowledge graph](Model_Recreation/knowledge_graph.png)

<br>
Work on KG at https://colab.research.google.com/drive/1TGTpVtqKdxZLKgZYcXwHTXeMIhkL2FBr
