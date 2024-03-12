import json
import yaml
import itertools

def dict_parser(x, parent_type=None):
    if type(x) is dict:
        children = [dict_parser(val,parent_type=dict) for val in x.values()]
        return [dict(zip(x.keys(), tup)) for tup in itertools.product(*children)]
    elif type(x) is list and parent_type!=tuple:
        return list(itertools.chain(*map(dict_parser, x)))
    elif type(x) is tuple:
        children = [dict_parser(val,parent_type=tuple) for val in x]
        return [tup for tup in itertools.product(*children)]
    else:
        return [x]

def parse(fname): 
    with open(fname) as f:
        extension = fname.split('.')[-1].lower()
        if extension == 'json':
            orig = json.loads(f.read())
        elif extension in ['yaml', 'yml']:
            orig = yaml.load(f, Loader = yaml.FullLoader)
        else:
            print("Config extension unknown:", extension)
            assert(False)
            
    return dict_parser(orig), orig