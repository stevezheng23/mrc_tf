import argparse
import json

import numpy as np

def add_arguments(parser):
    parser.add_argument("--input_file", help="path to input file", required=True)
    parser.add_argument("--output_file", help="path to output file", required=True)

def convert_quac(input_file,
                 output_file):
    with open(input_file, "r") as file:
        input_data = json.load(file)
    
    data_lookup = {}
    for data in input_data:
        qas_id = data["qas_id"]
        id_items = qas_id.split('#')
        id = id_items[0]
        turn_id = int(id_items[1])
        
        yes_no_list = ["y", "x", "n"]
        yes_no = yes_no_list[data["yes_no_id"]]
        
        follow_up_list = ["y", "m", "n"]
        follow_up = follow_up_list[data["follow_up_id"]]
        
        answer_text = data["predict_text"]
        
        if id not in data_lookup:
            data_lookup[id] = []
        
        data_lookup[id].append({
            "qas_id": qas_id,
            "turn_id": turn_id,
            "answer_text": answer_text,
            "yes_no": yes_no,
            "follow_up": follow_up
        })
    
    with open(output_file, "w") as file:
        for id in data_lookup.keys():
            data_list = sorted(data_lookup[id], key=lambda x: x["turn_id"])
            
            output_data = json.dumps({
                "best_span_str": [data["answer_text"] for data in data_list],
                "qid": [data["qas_id"] for data in data_list],
                "yesno": [data["yes_no"] for data in data_list],
                "followup": [data["follow_up"] for data in data_list]
            })
            
            file.write("{0}\n".format(output_data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    convert_quac(args.input_file, args.output_file)
