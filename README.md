# Vietnamese word tokenization use bert
The model achieved F1 score : 0.984

### Installation
```bash
git clone https://github.com/bino282/ViNLP.git
cd ViNLP
python setup.py develop build
```

### Test
``` bash
from ViNLP import BertVnTokenizer
tokenizer = BertVnTokenizer()
sentences = tokenizer.split(["Tổng thống Donald Trump ký sắc lệnh cấm mọi giao dịch của Mỹ với ByteDance và Tecent - chủ sở hữu của 2 ứng dụng phổ biến TikTok và WeChat sau 45 ngày nữa."])
print(sentences[0])
```

``` bash
Tổng_thống Donald_Trump ký sắc_lệnh cấm mọi giao_dịch của Mỹ với ByteDance và Tecent - chủ_sở_hữu của 2 ứng_dụng phổ_biến TikTok và WeChat sau 45 ngày nữa .

```

### Contact information
For personal communication related to this project, please contact Nha Nguyen Van (nha282@gmail.com).
