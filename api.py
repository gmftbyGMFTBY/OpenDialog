from header import *
from utils import *
from api_utils import *

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
        'from_id': data['robot_id'],
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
        table = init_mongodb('dialog', 'test')
        toUser, fromUser, msgType, content = parse_msg(request)
        
        # speical order
        if '#clear' == content:
            table.delete_many({})
            logger.info(f'[!] delete all the utterances in the database')
            return reply_text(fromUser, toUser, '#clear database over')
        elif content.startswith('#kg'):
            # kg-driven chat
            try:
                kg_path = eval(content[3:])
            except:
                return reply_text(fromUser, toUser, f'{content} error')
            session['kg_path'] = kg_path
            session['node'] = 0
            args['session'] = session
        
        table.insert_one({
            'toUser': toUser,
            'fromUser': fromUser,
            'utterance': content, 
            'id': db_table_counter(table),
        })
        args['table'], args['fromUser'], args['toUser'] = table, fromUser, toUser
        
        # chat and obtain the response
        reply = chat(agent, content, args=args, logger=logger)
        table.insert_one({
            'toUser': fromUser,
            'fromUser': toUser,
            'utterance': reply,
            'id': db_table_counter(table),
        })
        if args['verbose']:
            string = '\n========== LOG ==========\n'
            string += f'[Context] {args["content"]}\n[Response] {reply}'
            string += '\n========== LOG ==========\n'
            logger.info(string)
        return reply_text(fromUser, toUser, reply)

if __name__ == "__main__":
    args = vars(parser_args_api())
    agent = flask_load_agent(args['model'], args['gpu_id'], logger)
    app.run(host="0.0.0.0", port=8080)
