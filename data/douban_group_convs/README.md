[Douban Group Conversation 数据下载地址](https://github.com/gmftbyGMFTBY/douban_group_convs)

Douban Group Conversations is a corpus of online conversations crawled from the Chinese web forum Douban Group (https://www.douban.com/group). People discuss on a specific topic as a group. The messages of a given group form a conversation. The ground-truth of the reply-to relations of each conversation is obtained by tracking the quoting information of each message. This dataset contains 10,425 conversations with 137,980 messages. After performing Chinese word-cut, each message on average results in 12 words, which is very short.


When using this dataset in academic studies, you need to cite:
@INPROCEEDINGS{JunChen:dasfaa2017, 
	AUTHOR = {Jun Chen and Chaokun Wang and Heran Lin and Weiping Wang and Zhipeng Cai and Jianmin Wang}, 
	TITLE = {Learning the Structures of Online Asynchronous Conversations}, 
	BOOKTITLE = {Proceedings of International Conference on Database Systems for Advanced Applications}, 
	PAGES = {19-34}, 
	YEAR = "2017", }


Below shows the format of each of the three files in this dataset. With the following files, you can easily reconstruct the original conversations. If you have any question regarding this dataset, please contact the first author via E-mail: chenjun082@gmail.com. 


1. trees.json

A list of dict. Each dict shows the structure of a unique conversation which is shown below:
-------------------------------------------
	{
		'msg_1':{
					'msg_3':{
								...
							},
					'msg_4':None,
					...
				},
		'msg_2':{
					'msg_5':None,
					'msg_6':None,
					'msg_7':{
								'msg_8':{
											...
										}
							},
					...
				},
		...
	}
-------------------------------------------
Each 'key' is a message (represented by the line number of this message in corpus.jsonl). The reply-to relation is shown by the 'key:values' mapping. Each element in 'values' is a reply to 'key'. If one 'key' maps to None, it has no replies in this conversation (the leaf node). 


2. corpus.jsonl

Each line is a unique message. You need to separately parse each line with json. The structure of a line is a dict as below:
-------------------------------------------
	{
		"message":"XXX",
		"message_id": "YYY",
		"time": "yyyy-mm-ddTHH:MM:SSZ",
		"user": {
					"id":"ZZZ",
					"name":"AAA"
				}
	}
-------------------------------------------


3. tokens.json

The tokenized representations of messages with respect to corpus.jsonl. Each message has the same line numbers in both files. Each message is tokenized by performing Chinese word-cut with jieba (https://github.com/fxsjy/jieba). Then, each cutted word is represented by a unique integer. Thus, each message in tokens.json is represented by a list of integers (in word sequence).


