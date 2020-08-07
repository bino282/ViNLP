# Vietnamese word tokenization use bert
The model achieved F1 score : 0.984


``` bash
from ViNLP import BertVnTokenizer
tokenizer = BertVnTokenizer()
product_seg = tokenizer.split(["hôm nay tôi đi học"])
print(product_seg)

['hôm_nay tôi đi học']


```

### Contact information
For personal communication related to this project, please contact Nha Nguyen Van (nha282@gmail.com).
