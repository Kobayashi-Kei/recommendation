import requests
import re
import datetime
import lineNotifier
import traceback
import os
import time

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
header = {
    'User-Agent': user_agent
}
url = 'https://eplus.jp/sf/detail/3516430001'

try:
    while True:
        
        response = requests.get(url, headers=header)
        print(response.text)
        state = False
        if not '<span class="ticket-status__item ticket-status__item--accepting">予定枚数終了</span>' in  response.text:
            state = True

        dt_now = datetime.datetime.now()
        if state == True:
            msg = "Ticket found! " + dt_now.strftime('%Y/%m/%d/%H:%M:%S')
            print(msg, flush=True)
            lineNotifier.line_notify(msg)
        else:
            print("Not found " + dt_now.strftime('%Y/%m/%d/%H:%M:%S'), flush=True)
        
        time.sleep(30)
        
except Exception as e:
    print(traceback.format_exc())
    message = "E-pulus Error: " + \
        os.path.basename(__file__) + " " + str(traceback.format_exc())
    lineNotifier.line_notify(message)