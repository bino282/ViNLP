# from ViNLP import BertVnTokenizer
# tokenizer = BertVnTokenizer()
# sentences = tokenizer.tokenizer.tokenize("Tổng thống Donald Trump ký sắc lệnh cấm mọi giao dịch của Mỹ với ByteDance và Tecent")
# print(sentences)
from ViNLP import BertVnNer
bert_ner_model = BertVnNer()
sentence = "Về cơ chế đặc thù cho TP Hải Phòng, Thủ tướng cho biết Bộ Chính trị đã ban hành Nghị quyết 45 về phát triển TP Hải Phòng và Chính phủ đang có chương trình hành động để triển khai Nghị quyết này, đóng góp vào sự phát triển của thành phố Cảng, sánh vai cùng các thành phố của khu vực châu Á."
entities = bert_ner_model.annotate([sentence])
print(entities)