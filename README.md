# Vietnamese toolkit with bert

### Installation
```bash
git clone https://github.com/bino282/ViNLP.git
cd ViNLP
python setup.py develop build
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
sentence = "Về cơ chế đặc thù cho TP Hải Phòng, Thủ tướng cho biết Bộ Chính trị đã ban hành Nghị quyết 45 về phát triển TP Hải Phòng và Chính phủ đang có chương trình hành động để triển khai Nghị quyết này, đóng góp vào sự phát triển của thành phố Cảng, sánh vai cùng các thành phố của khu vực châu Á."
entities = bert_ner_model.annotate([sentence])
print(entities)

```
``` bash
[{'LOCATION': ['TP Hải Phòng', 'châu Á'], 'ORGANIZATION': ['Bộ Chính trị']}]

```




### Contact information
For personal communication related to this project, please contact Nha Nguyen Van (nha282@gmail.com).
