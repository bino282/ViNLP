# from ViNLP import BertVnTokenizer
# tokenizer = BertVnTokenizer()
# sentences = tokenizer.tokenizer.tokenize("Tổng thống Donald Trump ký sắc lệnh cấm mọi giao dịch của Mỹ với ByteDance và Tecent")
# print(sentences)
from ViNLP import BertVnNer
bert_ner_model = BertVnNer()
sentence = "Theo SCMP, báo cáo của CSIS với tên gọi Định hình Tương lai Chính sách của Mỹ với Trung Quốc cũng cho thấy sự ủng hộ tương đối rộng rãi của các chuyên gia về việc cấm Huawei, tập đoàn viễn thông khổng lồ của Trung Quốc"
entities = bert_ner_model.annotate([sentence])
print(entities)