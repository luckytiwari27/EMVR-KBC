import os
import pandas as pd
import pickle          
import time             

from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from groq import Groq
import time
from google.api_core.exceptions import ResourceExhausted

LLAMA_SOURCE="groq"

from data import convert_kgtxt_to_natlang_plus

def get_target_relation(subgraphs):
    return  convert_kgtxt_to_natlang_plus(subgraphs[0]["zerohop"][0][1])

def _triplet_to_input(triplet):
    return "({}, {}, {})".format(convert_kgtxt_to_natlang_plus(triplet[0]),convert_kgtxt_to_natlang_plus(triplet[1]),convert_kgtxt_to_natlang_plus(triplet[2]))

def _triplets_to_input(triplets):
    l = []
    for triplet in triplets:
        l.append(_triplet_to_input(triplet))
    return "\n".join(l)

def _get_subgraph_triplets(subgraph):
    triplets = []
    for level in subgraph:
        triplets += subgraph[level]
    return triplets

def _convert_subgraph_to_input(subgraph):
    target_fact = _triplet_to_input(subgraph["zerohop"][0])

    triplets = _get_subgraph_triplets(subgraph)
    related_subgraph = _triplets_to_input(triplets)
    llm_input_text = "A knowledge subgraph describes relationships between entities using a set of triplets. Each triplet is written in the form of triplet (SUBJ, REL, OBJ), which states that entity SUBJ is of relation REL to entity OBJ.\nA logic rule can be applied to known triplets to deduce new ones. Each rule is written in the form of a logical implication, which states that if the conditions on the right-hand side are satisfied, then the statement on the left-hand side holds true. Here are some example rules where A, B, C are entities:\nIF (A, parent, B) AND  NOT (A, father, B) THEN (A, mother, B)\nIF (A, father, B) OR (A, mother, B) THEN (A, parent, B)\nIF (A, mother, B) AND (A, sibling, C) THEN (C, mother, B)\nNow we have the following triplets:\n" + related_subgraph + "\nPlease generate as many of the most important logical rules based on the above knowledge subgraph to deduce triplet " + target_fact + ". The rules provide general logic implications instead of using specific entities. Return the rules only without any explanations."
    return llm_input_text


def convert_subgraphs_to_inputs(subgraphs,verbose=False):
    relation = get_target_relation(subgraphs)
    llm_inputs = []
    for subgraph in subgraphs:
        llm_input = _convert_subgraph_to_input(subgraph)
        llm_inputs.append(llm_input)
    if verbose: print("[{}] For relation {}, {} subgraphs compiled for LLM inputs".format(time.strftime("%Y-%m-%d %H:%M"), relation, len(llm_inputs)))
    return llm_inputs

def llm_input_len_check(llm_inputs, max_char=2048*4, verbose=True):
    llm_inputs_within_char_count = []
    char_counts = []
    for llm_input in llm_inputs:
        char_count = len(llm_input)
        char_counts.append(char_count)
        if char_count <= max_char:
            llm_inputs_within_char_count.append(llm_input)
    avg_char_count = sum(char_counts)/len(char_counts)
    if verbose:
        print("\taverage input char count: {}".format(avg_char_count),flush=True)

        max_char = max(char_counts)
        max_idx = char_counts.index(max(char_counts))
        print("\tthe current longest input of {} chars at index {}".format(max_char,max_idx),flush=True)
    return llm_inputs_within_char_count

def gpt35_set_api(api_key):
    client = OpenAI(api_key=api_key)
    return client

def gpt40_set_api(api_key):
    client = OpenAI(api_key=api_key)
    return client

def gemini_set_api(api_key):
    genai.configure(api_key=api_key)

def llama3_set_api(api_key):
    if LLAMA_SOURCE == "groq":
        client = Groq(api_key=api_key)
    else:
        raise Exception("invalid llama server")
    return client

def gpt35_proposing_rules(llm_inputs,client,verbose=True):
    if verbose:
        print("FUNCTION STARTED {}".format(time.strftime("%Y-%m-%d %H:%M")))
    proposed_rules = []
    completion_results = []
    counter, count_interval = 0, 100
    for llm_input in llm_inputs:
        if verbose and counter % count_interval == 0:
            print("\t {} / {} Done: {}".format(counter, len(llm_inputs), time.strftime("%Y-%m-%d %H:%M")))
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": llm_input,
                }
            ],model="gpt-3.5-turbo-0125")
        proposed_rules.append(chat_completion.choices[0].message.content)
        completion_results.append(chat_completion)
        counter += 1
    assert len(proposed_rules)==len(llm_inputs)
    if verbose:
        print("FUNCTION FINISHED: {}".format(time.strftime("%Y-%m-%d %H:%M")))
    return proposed_rules, completion_results

def gpt40_proposing_rules(llm_inputs,client,verbose=True):
    if verbose:
        print("FUNCTION STARTED {}".format(time.strftime("%Y-%m-%d %H:%M")))
    proposed_rules = []
    completion_results = []
    counter, count_interval = 0, 100
    for llm_input in llm_inputs:
        if verbose and counter % count_interval == 0:
            print("\t {} / {} Done: {}".format(counter, len(llm_inputs), time.strftime("%Y-%m-%d %H:%M")))
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": llm_input,
                }
            ],model="gpt-4-turbo-2024-04-09")
        proposed_rules.append(chat_completion.choices[0].message.content)
        completion_results.append(chat_completion)
        counter += 1
    assert len(proposed_rules)==len(llm_inputs)
    if verbose:
        print("FUNCTION FINISHED: {}".format(time.strftime("%Y-%m-%d %H:%M")))
    return proposed_rules, completion_results

def gemini15_proposing_rules(llm_inputs, model, verbose=True):
    if verbose:
        print("FUNCTION STARTED {}".format(time.strftime("%Y-%m-%d %H:%M")))
    proposed_rules = []
    completion_results = []
    counter, count_interval = 0, 100
    consecutive_rate_limit_hits = 0
    # for llm_input in llm_inputs:
    for llm_input in llm_inputs:
        if verbose and counter % count_interval == 0:
            print("\t {} / {} Done: {}".format(counter, len(llm_inputs), time.strftime("%Y-%m-%d %H:%M")))
        # response = model.generate_content(llm_input,
        #                                   safety_settings={
        #                                       HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        #                                       HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        #                                       HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        #                                       HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        #                                       })

        while True:
            try:
                response = model.generate_content(llm_input,
                                                safety_settings={
                                                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                                })

                # Small delay between successful requests
                time.sleep(30)
                consecutive_rate_limit_hits = 0
                break
            except ResourceExhausted as e:
                consecutive_rate_limit_hits += 1
                if consecutive_rate_limit_hits >= 3:
                    print("\n========================================")
                    print("Gemini rate limit hit 3 times in a row (~3+ minutes with no")
                    print("progress). This usually means the DAILY quota is exhausted, not")
                    print("just a per-minute limit -- retrying forever won't help.")
                    print("Stop this run (Ctrl+C), switch LLM_API_KEY to a different")
                    print("account, and rerun the SAME --run_proposer command -- it will")
                    print("resume from this exact relation, no work is lost.")
                    print("========================================\n")
                    raise SystemExit(
                        "Gemini daily quota likely exhausted -- switch API keys and rerun "
                        "the same --run_proposer command to resume."
                    )
                print("\n========================================")
                print("Gemini rate limit reached ({} in a row).".format(consecutive_rate_limit_hits))
                print("Waiting 60 seconds before retrying...")
                print("========================================\n")
                time.sleep(60)
        try:
            proposed_rule = response.text
            proposed_rules.append(proposed_rule)
            completion_results.append(response)
        except ValueError as err:
            print("Value error {} when proposing rule using input {}".format(err, llm_input))
            continue
        counter += 1 
    if verbose:
        print("FUNCTION FINISHED: {}".format(time.strftime("%Y-%m-%d %H:%M")))
    return proposed_rules, completion_results



def llama3_proposing_rules(llm_inputs,client,verbose=True):
    if verbose:
        print("FUNCTION STARTED {}".format(time.strftime("%Y-%m-%d %H:%M")))
    proposed_rules = []
    completion_results = []
    counter, count_interval = 0, 100
    for llm_input in llm_inputs:
        if verbose and counter % count_interval == 0:
            print("\t {} / {} Done: {}".format(counter, len(llm_inputs), time.strftime("%Y-%m-%d %H:%M")))
        if LLAMA_SOURCE == "replicate":
            output=[]
            for event in client.stream(
                "meta/meta-llama-3-70b-instruct",
                input={"prompt": llm_input}
            ):
                completion_results.append(event)
                if event.event.value=='done':
                    break
                else:
                    output.append(event.data)
            proposed_rules.append("".join(output))
        elif LLAMA_SOURCE == "groq":
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": llm_input,
                    }
                ], model="llama-3.3-70b-versatile")
            proposed_rule = chat_completion.choices[0].message.content
            proposed_rules.append(proposed_rule)
            completion_results.append(chat_completion)
        counter += 1
    assert len(proposed_rules)==len(llm_inputs)
    if verbose:
        print("FUNCTION FINISHED: {}".format(time.strftime("%Y-%m-%d %H:%M")))
    return proposed_rules, completion_results


def llm_propose_rule(llm_inputs, model_name, save_dir, save_pfx, api_key,save_pickle=True, verbose=True):
    if model_name in ["GPT35", "gpt35"]:
        client = gpt35_set_api(api_key)
        if verbose: print("trigger batch API call... ")
        proposed_rules, chat_results = gpt35_proposing_rules(llm_inputs,client, verbose=verbose)
        if verbose: print("batch API call completed. ")
    elif model_name in ["GPT40", "gpt40", "GPT4", "gpt4"]:
        client = gpt40_set_api(api_key)
        if verbose: print("trigger batch API call... ")
        proposed_rules, chat_results = gpt40_proposing_rules(llm_inputs,client, verbose=verbose)
        if verbose: print("batch API call completed. ")
    elif model_name in ["Gemini15", "gemini15"]:
        gemini_set_api(api_key)
        model = genai.GenerativeModel("gemini-1.5-flash-002")
        proposed_rules, chat_results = gemini15_proposing_rules(llm_inputs,model, verbose=verbose)
        if verbose:print("batch API call completed.")
    elif model_name in ["llama3", "llama"]:
        client = llama3_set_api(api_key)
        proposed_rules, chat_results = llama3_proposing_rules(llm_inputs, client, verbose=verbose)
        if verbose: print("batch API call completed. ")
    else:
        raise Exception("invalid model_name={}".format(model_name))
    if save_pickle:
        pickle_fname = os.path.join(save_dir, "{}_chat.pickle".format(save_pfx))
        with open(pickle_fname, 'wb') as handle:
            pickle.dump(chat_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose: print("chat results pickled into {}".format(pickle_fname))
        pickle_fname = os.path.join(save_dir, "{}_rule.pickle".format(save_pfx))
        with open(pickle_fname, 'wb') as handle:
            pickle.dump(proposed_rules, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose: print("proposed rules pickled into {}".format(pickle_fname))
    print("save_dir :", save_dir)
    print("save_pfx :", save_pfx)
    print("rawrules :", os.path.join(save_dir, "{}_proposed.csv".format(save_pfx)))
    rawrules_fname = os.path.join(save_dir, "{}_proposed.csv".format(save_pfx))
    proposed_df = pd.DataFrame(proposed_rules)
    proposed_df.to_csv(rawrules_fname)
    return proposed_rules