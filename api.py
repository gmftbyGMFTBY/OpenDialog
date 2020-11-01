from header import *
from utils import *
from api_utils import *
from config import *

'''API for wechat'''

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.route("/hello", methods=["GET"])
def hello():
    return 'hello'

# postman test API
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
        'timestamp': time.time()
    }
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

        token = api_args['token']    # your token

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
        table = init_mongodb(api_arg['mongodb']['database'], api_args['mongodb']['table'])
        toUser, fromUser, msgType, content = parse_msg(request)
        
        # speical order
        if api_args['special_cmd']['clear'] == content:
            table.delete_many({})
            logger.info(f'[!] delete all the utterances in the database')
            return reply_text(fromUser, toUser, '#clear database over')
        elif content.startswith(api_args['special_cmd']['kg']):
            # kg-driven chat
            try:
                kg_path = eval(content[3:])
            except:
                return reply_text(fromUser, toUser, f'{content} error')
            session['kg_path'], session['node'] = kg_path, 0
            api_args['session'], api_args['chat_mode'] = session, 2
            return reply_text(fromUser, toUser, f"#set the knowledge path {kg_path}")
        
        table.insert_one({
            'toUser': toUser,
            'fromUser': fromUser,
            'utterance': content, 
            'id': db_table_counter(table),
        })
        api_args['table'], api_args['fromUser'], api_args['toUser'] = table, fromUser, toUser
        
        # chat and obtain the response
        reply = chat(agent, content, args=api_args, logger=logger)
        table.insert_one({
            'toUser': fromUser,
            'fromUser': toUser,
            'utterance': reply,
            'id': db_table_counter(table),
        })
        if api_args['verbose']:
            string = '\n========== LOG ==========\n'
            string += f'[Context] {api_args["content"]}\n[Response] {reply}'
            string += '\n========== LOG ==========\n'
            logger.info(string)
        return reply_text(fromUser, toUser, reply)

if __name__ == "__main__":
    agent = flask_load_agent(api_args['model'], api_args['gpu_id'], logger)
    app.run(host=api_args['host'], port=api_args['port'])
