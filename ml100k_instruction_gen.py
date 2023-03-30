import time
import numpy as np
import json
import openai
import random
import argparse
import re
random.seed(2023)
np.random.seed(2023)
import time
import signal

api_keys = ["sk-",
            "sk-",
            "sk-",
            ] + ["sk-",
                 "sk-",
                 "sk-",
                 ] + ["sk-",
                      "sk-",
                      "sk-",
                      ]

api_key_idx = 0


def set_timeout(num, callback):
    def wrap(func):
        def handle(signum, frame):
            raise RuntimeError

        def to_do(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handle)
                signal.alarm(num)
                #                 print('start alarm signal.')
                r = func(*args, **kwargs)
                #                 print('close alarm signal.')
                signal.alarm(0)
                return r
            except RuntimeError as e:
                callback()

        return to_do

    return wrap


def after_timeout():
    #     print("Time out!")
    assert 1 == 0


@set_timeout(20, after_timeout)
def connect(messages, temperature=0):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=512,
        temperature=temperature,
    )
    return response


def call_func(messages, temperature=0):
    global api_key_idx
    openai.api_key = api_keys[api_key_idx]

    for ii in range(len(api_keys) * 2):
        try:
            openai.api_key = api_keys[api_key_idx]
            response = connect(messages)
            start_f = 0
        except:
            start_f = 1
            api_key_idx = (api_key_idx + 1) % len(api_keys)
        #             print (f'api_key_idx:{api_key_idx}')
        if start_f == 0:
            break

    return response["choices"][0]['message']['content']

# completion_with_backoff(model="text-davinci-002", prompt="Once upon a time,")
def read_json(file):
    with open(file) as f:
        return json.load(f)

def write_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

data_ml_100k = read_json("./ml_100k.json")

template_rec_instructs =[
    "Based on the movies that I've watched before, could you suggest some similar movies for me to watch next? Please use the MovieLens 100K dataset to recommend movies that you think would appeal to my tastes.",
    "I've seen a lot of movies that I really enjoyed, and I'm looking for recommendations on what to watch next. Can you suggest some similar films from the MovieLens 100K dataset that I might like?",
    "I'm looking for some new movies to watch that are similar to the ones I've enjoyed in the past. Using the MovieLens 100K dataset, could you suggest some titles that you think would be a good fit for me?"
]

instruction_data = []
for xx in range(len(data_ml_100k[:])):
    elem = data_ml_100k[xx]
    ground_truth = elem[-1]
    seq_list = elem[0].split(' | ')

    num_sublists = 5

    for i in range(num_sublists):

        instruct_per = {
            "instruction": "",
            "input": "",
            "output": ""
        }

        deno = len(seq_list) // 10 + 1
        sublist_length = random.randint(2, int(len(seq_list) / deno))
        start_index = random.randint(0, len(seq_list) - sublist_length)
        end_index = start_index + sublist_length
        sublist = seq_list[start_index:end_index]

        #         print (len(sublist), sublist)
        input_l = sublist[:-1]
        output_ = sublist[-1]
        #         print (f'input_l:{input_l}')
        #         print (f'output_:{output_}\n')

        if len(input_l) == 1:
            format_s = '{}'
            input_s = format_s.format(input_l[0])
            input_st = 'The movie I have watched is ' + input_s + "."
        elif len(input_l) == 2:
            format_s = '{} and {}'
            input_s = format_s.format(input_l[0], input_l[1])
            input_st = 'The movie I have watched are ' + input_s + "."
        else:
            format_s = '{}, and {}'
            input_s = format_s.format(', '.join(input_l[:-1]), input_l[-1])
            input_st = 'The movies I have watched are ' + input_s + "."

        template_rec_instruct = random.choice(template_rec_instructs)

        output_s = "One recommendation from the MovieLens 100K dataset is " + output_ + ". The recommendation reason is that "
        input_ss = template_rec_instruct + "\n" + input_s + "\n" + output_s
        #         print (input_ss, '\n')
        messages = [
            {"role": "user", "content": input_ss},
        ]
        predictions = call_func(messages, temperature=0.)

        instruct_per["instruction"] = template_rec_instruct
        instruct_per["input"] = input_s
        instruct_per["output"] = output_s + predictions

        instruction_data.append(instruct_per)
        print(f'{i} in {xx}')


    write_json(instruction_data, "ml_100k_instruct_data.json")
    print(f"{xx}/{len(data_ml_100k)}, saved")

