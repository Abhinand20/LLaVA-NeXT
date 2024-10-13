import argparse
import json

test_data = '/mnt/disks/data/webqa/WebQA_test.json'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()
    ans = {}
    with open(args.input_file, 'r') as f:
        for line in f:
            datum = line.strip().split('\t')
            ans[datum[0]] = datum[1]
    with open(test_data, 'r') as f:
        data = json.load(f)
    
    eval_data = {}
    for id,d in data.items():
        eval_data[id] = {}
        eval_data[id]['question'] = d['Q']
        eval_data[id]['sources'] = []
        eval_data[id]['answer'] = ans.get(id, d['Q']) 
    
    json.dump(eval_data, open('webqa_eval_fmt.json','w'))
    
    
if __name__ == "__main__":
    main()
