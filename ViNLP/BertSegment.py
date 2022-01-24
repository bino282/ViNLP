from ViNLP.BertPosTagger import BERTPoSTagger
from transformers import BertTokenizer
from transformers import  tokenization_bert
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import nltk
import torch
import re
import os
import unicodedata
import zipfile
import urllib.request
def _is_punctuation(char):
    return False
tokenization_bert._is_punctuation = _is_punctuation

class BertVnTokenizer:
    def __init__(self,model_path=None,max_length=256):
        if torch.cuda.is_available():       
            self.device = torch.device("cuda")

            print('There are %d GPU(s) available.' % torch.cuda.device_count())

            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")
        self.tag2int = {'B': 2, 'E': 3, 'I': 4, 'S': 5, '-PAD-': 0, '-SUB-': 1}
        self.int2tag = {2 : 'B',3 : 'E', 4 : 'I' , 5 : 'S', 0 : '-PAD-' , 1 : '-SUB-'}
        if(model_path is None): 
            path_root = os.path.join(os.path.expanduser('~'),".cache/torch/transformers")
            if(os.path.exists(os.path.join(path_root,"VnBertTokenizer"))):
                model_path = os.path.join(path_root,"VnBertTokenizer")
                print("Load model from cache : {}".format(os.path.join(path_root,"VnBertTokenizer")))
            else:
                if(not os.path.exists(path_root)):
                    os.makedirs(path_root)
                print("Downloading.... model from https://insai.s3-ap-southeast-1.amazonaws.com/transformers_model/VnBertTokenizer.zip") # this url is not valid now, please fix it
                urllib.request.urlretrieve('https://insai.s3-ap-southeast-1.amazonaws.com/transformers_model/VnBertTokenizer.zip', os.path.join(path_root,'VnBertTokenizer.zip'))
                print("Model is saved in {}".format(path_root))
                with zipfile.ZipFile(os.path.join(path_root,'VnBertTokenizer.zip'), 'r') as zip_ref:
                    zip_ref.extractall(path_root)
                model_path = os.path.join(path_root,'VnBertTokenizer')
        self.bertPoSTagger = BERTPoSTagger.from_pretrained(
                                            model_path,
                                            num_labels = len(self.tag2int),
                                            output_attentions = False,
                                            output_hidden_states = False,
                                    )
        
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False,tokenize_chinese_chars=False)
        self.MAX_LENGTH= max_length
        if torch.cuda.is_available():
            self.bertPoSTagger.cuda()

    def covert_text(self,texts,batch_size):
        self.sents_tok = []
        input_ids = []
        for i  in range(len(texts)):
            sent = texts[i]
            self.sents_tok.append(self.tokenizer.tokenize(sent)[0:self.MAX_LENGTH-2])
            encoded_sent = self.tokenizer.encode(
                                    sent,                      # Sentence to encode.
                                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                    #return_tensors = 'pt',     # Return pytorch tensors.
                        )
            if(len(encoded_sent)<= self.MAX_LENGTH):
                encoded_sent= encoded_sent + [0]*(self.MAX_LENGTH-len(encoded_sent))
            else:
                encoded_sent = encoded_sent[0:self.MAX_LENGTH-1] + [encoded_sent[-1]]
            input_ids.append(encoded_sent)

        attention_masks = []
        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)
        train_inputs = input_ids
        train_masks= attention_masks
        train_inputs = torch.tensor(train_inputs,dtype=torch.long)
        train_masks = torch.tensor(train_masks,dtype=torch.long)
        train_data = TensorDataset(train_inputs, train_masks)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        return train_dataloader

    def preprocess(self,texts):
        new_texts = []
        for t in texts:
            t = " ".join(nltk.word_tokenize(t))
            new_texts.append(t)
        return new_texts

    def predict(self,dataLoader):
        self.bertPoSTagger.eval()
        test_pred= []
        for step, batch in enumerate(dataLoader):
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask = batch
            with torch.no_grad():        
                outputs =  self.bertPoSTagger(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
                    
            predictions = outputs[0]
            max_preds = predictions.argmax(dim = 2, keepdim = True)
            lb = max_preds.cpu().detach().numpy().tolist()
            for m in range(len(lb)):
                sentence_new = ""
                sent_tok = self.sents_tok[step*self.batch_size+m]
                sent_ori = self.texts[step*self.batch_size+m].split()
                idx = 0
                for i in range(1,len(sent_tok)+1):
                    if ("##" in sent_tok[i-1]):
                        sentence_new = sentence_new + sent_tok[i-1].replace("##","")
                        continue
                    if(lb[m][i][0] in (5,2)):
                        if(sent_tok[i-1]=="[UNK]"):
                            sentence_new = sentence_new +" "+ sent_ori[idx]
                        else:
                            sentence_new = sentence_new +" "+ sent_tok[i-1]
                        idx = idx + 1
                    if(lb[m][i][0] in (4,3)):
                        if(sent_tok[i-1]=="[UNK]"):
                            sentence_new = sentence_new +"_"+ sent_ori[idx]
                            idx = idx + 1
                        else:
                            sentence_new = sentence_new + "_"+ sent_tok[i-1]
                            idx = idx + 1
                    if(lb[m][i][0]==1):
                        if(sent_tok[i-1]=="[UNK]"):
                            sentence_new = sentence_new + sent_ori[idx]
                            idx = idx + 1
                        else:
                            sentence_new = sentence_new + " "+sent_tok[i-1]
                            idx = idx +1
                test_pred.append(sentence_new.strip())
        return test_pred

    def split(self,texts,batch_size=16):
        texts = self.preprocess(texts)
        self.texts = texts
        self.batch_size = batch_size
        train_dataloader = self.covert_text(texts,batch_size)
        test_pred = self.predict(train_dataloader)
        return test_pred