import json
import os

def modelswitch(path,is_train = False):
    path = os.path.join(path,'config.json')
    with open(path,'r') as tp:
        data = json.load(tp)
    
    data['attn_mode'] = 'flashattn' if not is_train else 'flex'
    with open(path,'w') as wp:
        json.dump(data,wp)