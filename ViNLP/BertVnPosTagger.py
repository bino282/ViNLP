from ViNLP.BertPosTagger import BERTPoSTagger
from transformers import BertTokenizer
from transformers import  tokenization_bert
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import nltk
import torch
import re
import unicodedata
def _is_punctuation(char):
    return False
tokenization_bert._is_punctuation = _is_punctuation

class BertVnPosTagger:
    def __init__(self,model_path,max_length=256):
        self.tag2int = {'B_A': 2, 'B_Ab': 3, 'B_B': 4, 'B_C': 5, 'B_CH': 6, 'B_Cb': 7, 'B_Cc': 8, 'B_E': 9, 'B_Eb': 10, 'B_I': 11, 'B_L': 12, 'B_M': 13, 'B_Mb': 14, 'B_N': 15, 'B_Nb': 16, 'B_Nc': 17, 'B_Ni': 18, 'B_Np': 19, 'B_Nu': 20, 'B_Ny': 21, 'B_P': 22, 'B_Pb': 23, 'B_R': 24, 'B_T': 25, 'B_V': 26, 'B_Vb': 27, 'B_Vy': 28, 'B_X': 29, 'B_Xy': 30, 'B_Y': 31, 'B_Z': 32, 'I_A': 33, 'I_B': 34, 'I_C': 35, 'I_CH': 36, 'I_Cc': 37, 'I_E': 38, 'I_I': 39, 'I_L': 40, 'I_M': 41, 'I_N': 42, 'I_Nb': 43, 'I_Nc': 44, 'I_Np': 45, 'I_Nu': 46, 'I_Ny': 47, 'I_P': 48, 'I_R': 49, 'I_T': 50, 'I_V': 51, 'I_Vb': 52, 'I_X': 53, 'I_Y': 54, 'I_Z': 55, '-PAD-': 0, '-SUB-': 1}
        self.int2tag = {v: k for k, v in self.tag2int.items()}
        if torch.cuda.is_available():       
            self.device = torch.device("cuda")

            print('There are %d GPU(s) available.' % torch.cuda.device_count())

            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")
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
            lb_text = []
            for j in range(len(lb)):
                t = []
                for k in range(len(lb[j])):
                    t.append(self.int2tag[lb[j][k][0]])
                lb_text.append(t)
            for m in range(len(lb_text)):
                sentence_new = ""
                sent_tok = self.sents_tok[step*self.batch_size+m]
                sent_ori = self.texts[step*self.batch_size+m].split()
                idx = 0
                tag_s = []
                for i in range(1,len(sent_tok)+1):
                    if("_" in lb_text[m][i]):
                        w = lb_text[m][i].split("_")[0]
                        t = lb_text[m][i].split("_")[1]
                    else:
                        w = lb_text[m][i]
                        t = lb_text[m][i]
                    if ("##" in sent_tok[i-1]):
                        sentence_new = sentence_new + sent_tok[i-1].replace("##","")
                        continue
                    if(w == "B"):
                        if(sent_tok[i-1]=="[UNK]"):
                            sentence_new = sentence_new +" "+ sent_ori[idx]
                        else:
                            sentence_new = sentence_new +" "+ sent_tok[i-1]
                        tag_s.append(t)
                        idx = idx + 1
                    if(w == "I"):
                        if(sent_tok[i-1]=="[UNK]"):
                            sentence_new = sentence_new +"_"+ sent_ori[idx]
                            idx = idx + 1
                        else:
                            sentence_new = sentence_new + "_"+ sent_tok[i-1]
                            idx = idx + 1

                sentence_tag = []
                if(len(tag_s)!=len(sentence_new.split())):
                    print(tag_s)
                    print(sentence_new)
                    print(lb_text[m])
                for n in range(len(sentence_new.split())):
                    sentence_tag.append(sentence_new.split()[n]+"/"+tag_s[n])
                test_pred.append(" ".join(sentence_tag))
        return test_pred

    def split(self,texts,batch_size=16):
        texts = self.preprocess(texts)
        self.texts = texts
        self.batch_size = batch_size
        train_dataloader = self.covert_text(texts,batch_size)
        test_pred = self.predict(train_dataloader)
        return test_pred