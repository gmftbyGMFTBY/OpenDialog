from .header import *

'''
Hugging Face DialoGPT:
    https://huggingface.co/micrsoft/DialoGPT-medium
The DialoGPT is English version, but our repo faces the Chinese version.
So the simple Machine Translation API is used.
'''

class DialoGPTAgent(BaseAgent):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
        self.model = AutoModelWithLMHead.from_pretrained('microsoft/DialoGPT-medium')
        to_cuda(self.model, model=True)
        self.zh2en_api = 'http://translate.google.cn/translate_a/single?client=gtx&dt=t&dj=1&ie=UTF-8&sl=zh&tl=en&q='
        self.en2zh_api = 'http://translate.google.cn/translate_a/single?client=gtx&dt=t&dj=1&ie=UTF-8&sl=en&tl=zh&q='

    def translate(self, msgs, tgt_lang='zh'):
        '''
        translate the chinese into english for input;
        translate the english into chinese for output;
        '''
        api = self.en2zh_api if tgt_lang == 'zh' else self.zh2en_api
        data = []
        for u in msgs.split('[SEP]'):
            u = u.strip()
            api_q = f'{api}{u}'
            data.append(eval(requests.get(api_q).text)['sentences'][0]['trans'])
            data.append(' [SEP] ')
        data = ''.join(data[:-1])
        return data

    def talk(self, topic, msgs):
        msgs = self.translate(msgs, tgt_lang='en')
        msgs = f'>> User: {msgs}'
        msgs = self.tokenizer.encode(msgs + self.tokenizer.eos_token, return_tensors='pt')
        msgs = to_cuda(msgs)
        generated = self.model.generate(
                msgs, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        ipdb.set_trace()
        tgt = self.tokenizer.decode(generated[:, msgs.shape[-1]:][0], skip_special_tokens=True)
        tgt = self.translate(tgt, tgt_lang='zh')
        return tgt 

if __name__ == "__main__":
    agent = DialoGPTAgent()
    agent.talk(None, '我觉得你是一个非常有自信的人。真的吗？')
