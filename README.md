# Vietnamese toolkit with bert
ViNLP is a system annotation for Vietnamese, it use pretrain [Bert4news](https://github.com/bino282/bert4news/) to fine-turning to NLP problems in Vietnamese components of wordsegmentation,Named entity recognition (NER)  and achieve high accuracy.

New version of tookit use pretrained model NlpHUST/electra-base-vn with higher accuaracy and easier to use at [NlpHUST/ner-vietnamese-electra-base](https://huggingface.co/NlpHUST/ner-vietnamese-electra-base?text=Li%C3%AAn+quan+v%E1%BB%A5+vi%E1%BB%87c+CSGT+b%E1%BB%8B+t%E1%BB%91+%C4%91%C3%A1nh+d%C3%A2n%2C+tr%C3%BAng+m%E1%BB%99t+ch%C3%A1u+nh%E1%BB%8F+%C4%91ang+ng%E1%BB%A7%2C+%C4%91ang+lan+truy%E1%BB%81n+tr%C3%AAn+m%E1%BA%A1ng+x%C3%A3+h%E1%BB%99i%2C+%C4%90%E1%BA%A1i+t%C3%A1+Nguy%E1%BB%85n+V%C4%83n+T%E1%BA%A3o%2C+Ph%C3%B3+Gi%C3%A1m+%C4%91%E1%BB%91c+C%C3%B4ng+an+t%E1%BB%89nh+Ti%E1%BB%81n+Giang+v%E1%BB%ABa+c%C3%B3+cu%E1%BB%99c+h%E1%BB%8Dp+c%C3%B9ng+Ch%E1%BB%89+huy+C%C3%B4ng+an+huy%E1%BB%87n+Ch%C3%A2u+Th%C3%A0nh+v%C3%A0+m%E1%BB%99t+s%E1%BB%91+%C4%91%C6%A1n+v%E1%BB%8B+nghi%E1%BB%87p+v%E1%BB%A5+c%E1%BA%A5p+t%E1%BB%89nh+%C4%91%E1%BB%83+ch%E1%BB%89+%C4%91%E1%BA%A1o+l%C3%A0m+r%C3%B5+th%C3%B4ng+tin.)

### Installation
```bash
git clone https://github.com/bino282/ViNLP.git
cd ViNLP
python setup.py develop build
```
or 
```bash
pip install git+https://github.com/bino282/ViNLP.git
```

### Test Segmentation
The model achieved F1 score : 0.984 on VLSP 2013 dataset

|Model | F1 |
|--------|-----------|
| **BertVnTokenizer** | 98.40 |
| **DongDu** | 96.90 |
| **JvnSegmenter-Maxent** | 97.00 |
| **JvnSegmenter-CRFs** | 97.06 |
| **VnTokenizer** | 97.33 |
| **UETSegmenter** | 97.87 |
| **VnTokenizer** | 97.33 |
| **VnCoreNLP (i.e. RDRsegmenter)** | 97.90 |


``` bash
from ViNLP import BertVnTokenizer
tokenizer = BertVnTokenizer()
sentences = tokenizer.split(["Tổng thống Donald Trump ký sắc lệnh cấm mọi giao dịch của Mỹ với ByteDance và Tecent - chủ sở hữu của 2 ứng dụng phổ biến TikTok và WeChat sau 45 ngày nữa."])
print(sentences[0])
```
``` bash
Tổng_thống Donald_Trump ký sắc_lệnh cấm mọi giao_dịch của Mỹ với ByteDance và Tecent - chủ_sở_hữu của 2 ứng_dụng phổ_biến TikTok và WeChat sau 45 ngày nữa .

```

### Test Named Entity Recognition
The model achieved F1 score VLSP 2018 for all named entities including nested entities : 0.786

|Model | F1 |
|--------|-----------|
| **BertVnNer** | 78.60 |
| **VNER Attentive Neural Network** | 77.52 |
| **vietner CRF (ngrams + word shapes + cluster + w2v)** | 76.63 |
| **ZA-NER BiLSTM** | 74.70 |

``` bash
from ViNLP import BertVnNer
bert_ner_model = BertVnNer()
sentence = "Theo SCMP, báo cáo của CSIS với tên gọi Định hình Tương lai Chính sách của Mỹ với Trung Quốc cũng cho thấy sự ủng hộ tương đối rộng rãi của các chuyên gia về việc cấm Huawei, tập đoàn viễn thông khổng lồ của Trung Quốc"
entities = bert_ner_model.annotate([sentence])
print(entities)

```
``` bash
[{'ORGANIZATION': ['SCMP', 'CSIS', 'Huawei'], 'LOCATION': ['Mỹ', 'Trung Quốc']}]

```




### Contact information
For personal communication related to this project, please contact Nha Nguyen Van (nha282@gmail.com).
