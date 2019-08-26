import argparse
import json

import numpy as np

def add_arguments(parser):
    parser.add_argument("--input_file", help="path to input file", required=True)
    parser.add_argument("--output_file", help="path to output file", required=True)

def convert_coqa(input_file,
                 output_file):
    with open(input_file, "r") as file:
        input_data = json.load(file)
    
    output_data = []
    for data in input_data:
        id_items = data["qas_id"].split('_')
        id = id_items[0]
        turn_id = int(id_items[1])
        
        answer_id = data["answer_id"]
        answer_score = data["answer_score"]
        
        answer_text_list = [data["predict_text"], "unknown", "yes", "no"]
        answer_text = answer_text_list[answer_id]
        
        output_data.append({
            "id": id,
            "turn_id": turn_id,
            "answer": answer_text,
            "score": answer_score
        })
    
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    convert_coqa(args.input_file, args.output_file)
