import argparse
import json

import numpy as np

def add_arguments(parser):
    parser.add_argument("--input_file", help="path to input file", required=True)
    parser.add_argument("--output_file", help="path to output file", required=True)
    parser.add_argument("--prob_threshold", help="probability threshold", required=False, default=0.1, type=float)

def convert_coqa(input_file,
                 output_file,
                 prob_threshold):
    with open(input_file, "r") as file:
        input_data = json.load(file)
    
    output_data = []
    for data in input_data:
        id_items = data["qas_id"].split('_')
        id = id_items[0]
        turn_id = int(id_items[1])
        
        prob_list = [
            data["unk_answer_prob"],
            data["yes_answer_prob"],
            data["no_answer_prob"]
        ]
        
        prob_idx = np.argmax(prob_list)
        
        answer = data["predict_text"]
        if prob_list[prob_idx] > prob_threshold:
            if prob_idx == 0:
                answer = "unknown"
            elif prob_idx == 1:
                answer = "yes"
            elif prob_idx == 2:
                answer = "no"
        
        output_data.append({
            "id": id,
            "turn_id": turn_id,
            "answer": answer
        })
    
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    convert_coqa(args.input_file, args.output_file, args.prob_threshold)
