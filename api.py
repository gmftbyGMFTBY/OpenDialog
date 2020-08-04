from header import *
from utils import *
from models import *
import sys

'''
API for SMP-MCC 2020 and wechat
'''

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
    
# init the agent
args = {}
args['model'] = sys.argv[1]
args['multi_gpu'] = sys.argv[2]
logger.info(f'[!] begin to init the {args["model"]} agent on {args["multi_gpu"]} GPU')
if args['model'] == 'bertretrieval':
    agent = BERTRetrievalAgent(args['multi_gpu'], kb=False)
    agent.load_model(f'ckpt/zh50w/bertretrieval/best.pt')
elif args['model'] == 'bertretrieval_multiview':
    agent = BERTMULTIVIEWAgent(args['multi_gpu'], kb=False)
    agent.load_model(f'ckpt/zh50w/bertretrieval_multiview/best.pt')
elif args['model'] == 'gpt2':
    # available run_mode: test, rerank, rerank_ir
    agent = GPT2Agent(1000, args['multi_gpu'], run_mode='rerank_ir')
    agent.load_model(f'ckpt/train_generative/gpt2/best.pt')
elif args['model'] == 'when2talk':
    agent = When2TalkAgent(1000, args['multi_gpu'], run_mode='test')
    agent.load_model(f'ckpt/when2talk/when2talk/best.pt')
elif args['model'] == 'test':
    agent = TestAgent()
elif args['model'] == 'multiview':
    agent = MultiViewTestAgent()
else:
    print(f'[!] obtain the unknown model name {args["model"]}')
    exit()
print(f'[!] init the {args["model"]} agent on GPU {args["multi_gpu"]} over')

@app.route("/hello", methods=["GET"])
def hello():
    return 'hello'

# SMP-MCC test API
@app.route('/get_res', methods=["POST", "GET"])
def get_res():
    '''
    data = {
        'group_id': group_id,
        'topic': topic,
        'robot_id': your_robot_id,
        'msgs': [
            {
                'from_id': robot_id,
                'msg': msg,
                'timestamp': timestamp
            }
        ]
    }
    '''
    data = request.json
    msg = agent.get_res(data)

    res = {
        'msg': msg,
        'from_id': data['robot_id'],
        'timestamp': time.time()
    }
    # show the log
    msgs_str = ' [SEP] '.join([i['msg'] for i in data['msgs']])
    log_str = f'\n========== LOG ==========\n[Context] {msgs_str}\n[Response] {msg}\n========== LOG ==========\n'
    logger.info(log_str)
    return jsonify(res)

# wechat api
@app.route('/wx/', methods=['GET', 'POST'])
def wechat_api():
    '''
    During talking, use the pymongo to save the conversation context
    Mongodb: dbname[dialog], table_name[test]
    '''
    if request.method == 'GET':
        # verify for wechat
        my_signature = request.args.get('signature')
        my_timestamp = request.args.get('timestamp')
        my_nonce = request.args.get('nonce')
        my_echostr = request.args.get('echostr')

        token = 'gmftbyGMFTBY'    # your token

        data = [token, my_timestamp, my_nonce]
        data.sort()

        temp = ''.join(data)
        sha1 = hashlib.sha1(temp.encode('utf-8'))
        hashcode = sha1.hexdigest()
        if my_signature == hashcode:
            return make_response(my_echostr)
        else:
            return make_response('')
    else:
        # POST method, talk to the chatbot
        table = init_mongodb('dialog', 'test')
        data = request.data.decode()
        xml = ET.fromstring(data)
        toUser = xml.find('ToUserName').text
        fromUser = xml.find('FromUserName').text
        msgType = xml.find('MsgType').text
        content = xml.find('Content').text
        # save the query into the database
        # special command of the user
        if content.startswith('#'):
            if content == '#清空':
                # clear all the conversation context in the database
                table.delete_many({})
                logger.info(f'[!] delete all the conversation in the database')
                return reply_text(fromUser, toUser, '#清空会话成功')
            else:
                # #体育, #电影, #数码产品, #食物, #音乐
                rest, topic = save_topic(content)
                if not rest:
                    return reply_text(fromUser, toUser, f'#未知的会话主题: {topic}')
                else:
                    return reply_text(fromUser, toUser, f'#设定会话主题为: {topic}')
        # load the topic
        topic = load_topic() 
        data = {'response': content, 'id': db_table_counter(table)}
        x = table.insert_one(data)
        logger.info(f'[!] insert the utterance {data["id"]} into the databse')
        # get response from the agent
        # obtain the recent conversation utterances
        # msgs = [(i['response'], i['id']) for i in table.find({})]
        # sort and only use 10 current utterances
        # msgs = sorted(msgs, key=lambda i:i[1])[-10:]
        data = {
            'group_id': 0,
            'topic': None,
            'robot_id': 0,
            'msgs': [{'msg': content}]
        }
        reply = agent.get_res(data)
        # insert the response into the mongodb
        data_ = {'response': reply, "id": db_table_counter(table)}
        x = table.insert_one(data_)
        logger.info(f'[!] insert the utterance {data_["id"]} into the databse')
        # show the log
        msgs_str = ' [SEP] '.join([i['msg'] for i in data['msgs']])
        log_str = f'\n========== LOG ==========\n[Context] {msgs_str}\n[Response] {reply}\n========== LOG ==========\n'
        logger.info(log_str)
        # return reply_text(fromUser, toUser, f'CTX: {content}\nTGT: {reply}')
        return reply_text(fromUser, toUser, f'{reply}')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
