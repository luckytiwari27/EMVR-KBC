"""
proposer.py -- LLM Proposer stage.

CHANGES vs the original version
-------------------------------
1. GEMINI MODEL IS NO LONGER HARDCODED. The old code pinned a single model id
   ("gemini-1.5-flash-002", later edited to "gemini-2.5-flash"). Google now
   returns 404 "no longer available to new users" for 2.x models on API keys
   with no prior usage of them, ahead of the 2026-10-16 shutdown. This version
   asks the API which models the key can actually reach (list_models) and picks
   the best one from a preference list. When Google retires another generation,
   this keeps working with no code edit.

   Override order:  GEMINI_MODEL env var  >  auto-discovery  >  hard fallback.

2. RATE-LIMIT HANDLING. The old code slept a flat 30 s after EVERY successful
   call. At ~24 subgraphs x 46 relations that is 9+ hours of pure sleeping on
   UMLS alone. Now: no sleep by default, exponential backoff only when the API
   actually pushes back. Tune with GEMINI_SLEEP if your quota is tight.

3. ALIGNMENT BUG. On a blocked/empty response the old loop hit `continue`
   before `counter += 1`, so the progress counter stalled and the returned rule
   list silently drifted out of alignment with llm_inputs. Failures now append
   an empty string, keeping index i of the output matched to index i of the
   input, and are counted and reported at the end.

4. Clear, actionable errors for 404 (model gone) and 403/401 (bad key), instead
   of a raw gRPC traceback.

Public API is unchanged: lesr.py calls llm_propose_rule(...) exactly as before.
"""

import os
import pickle
import re
import time

import pandas as pd

from openai import OpenAI
import google.generativeai as genai
from groq import Groq
from groq import RateLimitError as GroqRateLimitError
from google.api_core.exceptions import (
    ResourceExhausted, NotFound, PermissionDenied, Unauthenticated,
    ServiceUnavailable, InternalServerError, DeadlineExceeded,
)

try:  # safety-setting enums moved around between SDK versions
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    _SAFETY = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
except Exception:
    _SAFETY = None

LLAMA_SOURCE = "groq"

from data import convert_kgtxt_to_natlang_plus


# ==========================================================================
# Gemini model resolution
# ==========================================================================

# Newest first. Discovery walks this list and takes the first entry the key can
# actually call. Add new ids at the top as Google ships them.
GEMINI_MODEL_PREFERENCE = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash",
    "gemini-2.5-flash",          # blocked for new keys; kept as a last resort
    "gemini-2.5-flash-lite",
]

# Seconds to sleep after a SUCCESSFUL call. 0 = no throttle (default).
# Raise via env var if you are on a tight free-tier RPM budget.
GEMINI_SLEEP = float(os.environ.get("GEMINI_SLEEP", "0"))

_RESOLVED_GEMINI_MODEL = None      # cache; discovery costs one API call


def _list_available_gemini_models():
    """Model ids the current key can call generateContent on, short names."""
    available = []
    for m in genai.list_models():
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            available.append(m.name.split("/", 1)[-1])
    return available


def resolve_gemini_model(verbose=True):
    """
    Pick a Gemini model this API key can actually use.

    GEMINI_MODEL env var wins outright (no discovery, no validation) so you can
    force a specific id. Otherwise we ask the API what is available and take the
    highest-preference match, falling back to any flash model it offers.
    """
    global _RESOLVED_GEMINI_MODEL

    forced = os.environ.get("GEMINI_MODEL", "").strip()
    if forced:
        if verbose:
            print("[proposer] GEMINI_MODEL is set -- using '{}' without discovery.".format(forced))
        _RESOLVED_GEMINI_MODEL = forced
        return forced

    if _RESOLVED_GEMINI_MODEL:
        return _RESOLVED_GEMINI_MODEL

    try:
        available = _list_available_gemini_models()
    except Exception as err:
        fallback = GEMINI_MODEL_PREFERENCE[1]
        print("[proposer] WARNING: could not list models ({}: {}). "
              "Falling back to '{}'.".format(type(err).__name__, err, fallback))
        _RESOLVED_GEMINI_MODEL = fallback
        return fallback

    # exact match against the preference list, best first
    for want in GEMINI_MODEL_PREFERENCE:
        if want in available:
            _RESOLVED_GEMINI_MODEL = want
            if verbose:
                print("[proposer] Gemini model auto-selected: '{}' "
                      "({} models available to this key).".format(want, len(available)))
            return want

    # nothing from the preference list -- take any non-preview flash model
    flash = [m for m in available
             if "flash" in m and "preview" not in m and "exp" not in m
             and "tts" not in m and "audio" not in m and "image" not in m]
    if flash:
        flash.sort(reverse=True)          # crude but works: 3.7 > 3.6 > 3.5
        _RESOLVED_GEMINI_MODEL = flash[0]
        print("[proposer] No preferred model available; using '{}'.".format(flash[0]))
        print("[proposer] Consider adding it to GEMINI_MODEL_PREFERENCE in proposer.py.")
        return flash[0]

    raise RuntimeError(
        "No Gemini model supporting generateContent is available to this API key.\n"
        "Models the key CAN see: {}\n"
        "Set GEMINI_MODEL to one of them, or check the key at "
        "https://aistudio.google.com/apikey".format(available[:20])
    )


def print_available_gemini_models(api_key):
    """Diagnostic helper. Run:
       python -c "import proposer; proposer.print_available_gemini_models('YOUR_KEY')"
    """
    genai.configure(api_key=api_key)
    models = sorted(_list_available_gemini_models())
    picked = resolve_gemini_model(verbose=False)
    print("{} models support generateContent with this key:".format(len(models)))
    for m in models:
        print("   {}{}".format(m, "   <-- would be selected" if m == picked else ""))


# ==========================================================================
# Prompt construction (unchanged from the original)
# ==========================================================================

def get_target_relation(subgraphs):
    return convert_kgtxt_to_natlang_plus(subgraphs[0]["zerohop"][0][1])


def _triplet_to_input(triplet):
    return "({}, {}, {})".format(
        convert_kgtxt_to_natlang_plus(triplet[0]),
        convert_kgtxt_to_natlang_plus(triplet[1]),
        convert_kgtxt_to_natlang_plus(triplet[2]))


def _triplets_to_input(triplets):
    return "\n".join(_triplet_to_input(t) for t in triplets)


def _get_subgraph_triplets(subgraph):
    triplets = []
    for level in subgraph:
        triplets += subgraph[level]
    return triplets


def _convert_subgraph_to_input(subgraph):
    target_fact = _triplet_to_input(subgraph["zerohop"][0])
    related_subgraph = _triplets_to_input(_get_subgraph_triplets(subgraph))
    llm_input_text = (
        "A knowledge subgraph describes relationships between entities using a set of triplets. "
        "Each triplet is written in the form of triplet (SUBJ, REL, OBJ), which states that entity "
        "SUBJ is of relation REL to entity OBJ.\n"
        "A logic rule can be applied to known triplets to deduce new ones. Each rule is written in "
        "the form of a logical implication, which states that if the conditions on the right-hand "
        "side are satisfied, then the statement on the left-hand side holds true. Here are some "
        "example rules where A, B, C are entities:\n"
        "IF (A, parent, B) AND  NOT (A, father, B) THEN (A, mother, B)\n"
        "IF (A, father, B) OR (A, mother, B) THEN (A, parent, B)\n"
        "IF (A, mother, B) AND (A, sibling, C) THEN (C, mother, B)\n"
        "Now we have the following triplets:\n" + related_subgraph +
        "\nPlease generate as many of the most important logical rules based on the above knowledge "
        "subgraph to deduce triplet " + target_fact +
        ". The rules provide general logic implications instead of using specific entities. "
        "Return the rules only without any explanations.")
    return llm_input_text


def convert_subgraphs_to_inputs(subgraphs, verbose=False):
    relation = get_target_relation(subgraphs)
    llm_inputs = [_convert_subgraph_to_input(sg) for sg in subgraphs]
    if verbose:
        print("[{}] For relation {}, {} subgraphs compiled for LLM inputs".format(
            time.strftime("%Y-%m-%d %H:%M"), relation, len(llm_inputs)))
    return llm_inputs


def llm_input_len_check(llm_inputs, max_char=2048 * 4, verbose=True):
    kept, char_counts = [], []
    for llm_input in llm_inputs:
        n = len(llm_input)
        char_counts.append(n)
        if n <= max_char:
            kept.append(llm_input)
    if verbose and char_counts:
        print("\taverage input char count: {}".format(sum(char_counts) / len(char_counts)), flush=True)
        print("\tthe current longest input of {} chars at index {}".format(
            max(char_counts), char_counts.index(max(char_counts))), flush=True)
    return kept


# ==========================================================================
# API clients
# ==========================================================================

def gpt35_set_api(api_key):
    return OpenAI(api_key=api_key)


def gpt40_set_api(api_key):
    return OpenAI(api_key=api_key)


def gemini_set_api(api_key):
    genai.configure(api_key=api_key)


def llama3_set_api(api_key):
    if LLAMA_SOURCE == "groq":
        return Groq(api_key=api_key)
    raise Exception("invalid llama server")


# ==========================================================================
# Rule proposing
# ==========================================================================

def _openai_proposing_rules(llm_inputs, client, model_id, verbose=True):
    if verbose:
        print("FUNCTION STARTED {}  (model={})".format(time.strftime("%Y-%m-%d %H:%M"), model_id))
    proposed_rules, completion_results = [], []
    count_interval = 100
    for counter, llm_input in enumerate(llm_inputs):
        if verbose and counter % count_interval == 0:
            print("\t {} / {} Done: {}".format(counter, len(llm_inputs), time.strftime("%Y-%m-%d %H:%M")))
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": llm_input}], model=model_id)
        proposed_rules.append(chat_completion.choices[0].message.content)
        completion_results.append(chat_completion)
    assert len(proposed_rules) == len(llm_inputs)
    if verbose:
        print("FUNCTION FINISHED: {}".format(time.strftime("%Y-%m-%d %H:%M")))
    return proposed_rules, completion_results


def gpt35_proposing_rules(llm_inputs, client, verbose=True):
    return _openai_proposing_rules(llm_inputs, client, "gpt-3.5-turbo-0125", verbose)


def gpt40_proposing_rules(llm_inputs, client, verbose=True):
    return _openai_proposing_rules(llm_inputs, client, "gpt-4-turbo-2024-04-09", verbose)


def _gemini_generate_once(model, llm_input):
    """One call. Tries with safety settings; retries without if the SDK/model
    rejects that argument (the enum set changed across Gemini generations)."""
    if _SAFETY is not None:
        try:
            return model.generate_content(llm_input, safety_settings=_SAFETY)
        except TypeError:
            pass
        except ValueError:
            pass
    return model.generate_content(llm_input)


def gemini_proposing_rules(llm_inputs, model, verbose=True,
                           max_retries=6, sleep_between=None):
    """
    Rate-limit strategy: no fixed delay on success. On ResourceExhausted (429),
    back off 30 s, 60 s, 120 s, 240 s ... up to max_retries. Six consecutive
    failures on one prompt means the DAILY quota is gone, not a per-minute cap,
    so we stop with instructions rather than spinning forever.
    """
    if sleep_between is None:
        sleep_between = GEMINI_SLEEP

    if verbose:
        print("FUNCTION STARTED {}".format(time.strftime("%Y-%m-%d %H:%M")))

    proposed_rules, completion_results = [], []
    count_interval, n_blocked = 100, 0

    for counter, llm_input in enumerate(llm_inputs):
        if verbose and counter % count_interval == 0:
            print("\t {} / {} Done: {}".format(counter, len(llm_inputs), time.strftime("%Y-%m-%d %H:%M")))

        response, attempt = None, 0
        while True:
            try:
                response = _gemini_generate_once(model, llm_input)
                break

            except ResourceExhausted:
                attempt += 1
                if attempt >= max_retries:
                    print("\n" + "=" * 70)
                    print("Gemini rate limit hit {} times in a row on a single prompt.".format(attempt))
                    print("That is almost always the DAILY quota, not a per-minute cap --")
                    print("retrying further will not help.")
                    print("")
                    print("Switch --llm_api_key to a different account and rerun the SAME")
                    print("--run_proposer command. Completed relations are skipped, so you")
                    print("resume from exactly this relation and lose no finished work.")
                    print("=" * 70 + "\n")
                    raise SystemExit("Gemini daily quota exhausted -- switch API keys and rerun.")
                wait = 30 * (2 ** (attempt - 1))
                print("[proposer] rate limited (attempt {}/{}), waiting {}s...".format(
                    attempt, max_retries, wait), flush=True)
                time.sleep(wait)

            except NotFound as err:
                raise SystemExit(
                    "\nGemini model not available to this API key:\n  {}\n\n"
                    "Google blocks older models for keys with no prior usage of them.\n"
                    "See which models your key CAN use:\n"
                    '  python -c "import proposer; proposer.print_available_gemini_models(\'YOUR_KEY\')"\n'
                    "Then either add that id to GEMINI_MODEL_PREFERENCE in proposer.py,\n"
                    "or force it for this run:   $env:GEMINI_MODEL = \"gemini-3.6-flash\"\n".format(err))

            except (PermissionDenied, Unauthenticated) as err:
                raise SystemExit(
                    "\nGemini rejected the API key ({}).\n"
                    "A valid AI Studio key starts with 'AIza'. A value starting with 'AQ.' is an\n"
                    "OAuth access token, which expires after about an hour.\n"
                    "Get a durable key at https://aistudio.google.com/apikey\n".format(
                        type(err).__name__))

            except (ServiceUnavailable, InternalServerError, DeadlineExceeded) as err:
                attempt += 1
                if attempt >= max_retries:
                    print("[proposer] giving up on prompt {} after {} transient errors: {}".format(
                        counter, attempt, err))
                    response = None
                    break
                wait = 10 * attempt
                print("[proposer] transient error ({}), retry {}/{} in {}s...".format(
                    type(err).__name__, attempt, max_retries, wait), flush=True)
                time.sleep(wait)

        # Keep output index-aligned with input: a failure contributes "".
        text = ""
        if response is not None:
            try:
                text = response.text
            except (ValueError, AttributeError) as err:
                n_blocked += 1
                if verbose:
                    print("[proposer] prompt {} produced no usable text ({}); "
                          "recording empty result.".format(counter, err))
        else:
            n_blocked += 1

        proposed_rules.append(text)
        completion_results.append(response)

        if sleep_between:
            time.sleep(sleep_between)

    assert len(proposed_rules) == len(llm_inputs)
    if verbose:
        print("FUNCTION FINISHED: {}  ({} of {} prompts returned no text)".format(
            time.strftime("%Y-%m-%d %H:%M"), n_blocked, len(llm_inputs)))
    return proposed_rules, completion_results


# Backwards-compatible alias: older code/patches may import this name.
gemini15_proposing_rules = gemini_proposing_rules


def llama3_proposing_rules(llm_inputs, client, verbose=True):
    if verbose:
        print("FUNCTION STARTED {}".format(time.strftime("%Y-%m-%d %H:%M")))
    proposed_rules, completion_results = [], []
    count_interval = 100
    consecutive_rate_limit_hits = 0
    for counter, llm_input in enumerate(llm_inputs):
        if verbose and counter % count_interval == 0:
            print("\t {} / {} Done: {}".format(counter, len(llm_inputs), time.strftime("%Y-%m-%d %H:%M")))
        if LLAMA_SOURCE == "replicate":
            output = []
            for event in client.stream("meta/meta-llama-3-70b-instruct", input={"prompt": llm_input}):
                completion_results.append(event)
                if event.event.value == 'done':
                    break
                output.append(event.data)
            proposed_rules.append("".join(output))
        elif LLAMA_SOURCE == "groq":
            while True:
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": llm_input}],
                        model="openai/gpt-oss-120b")
                    proposed_rules.append(chat_completion.choices[0].message.content)
                    completion_results.append(chat_completion)
                    consecutive_rate_limit_hits = 0
                    break
                except GroqRateLimitError as e:
                    consecutive_rate_limit_hits += 1
                    # Groq's TPM/RPM limits are per-minute rolling windows, so a
                    # short wait almost always clears it -- try to honour the
                    # exact wait time the API suggests ("try again in 472.5ms"),
                    # falling back to a flat few seconds if that can't be parsed.
                    wait_seconds = 5.0
                    match = re.search(r"try again in ([\d.]+)(ms|s)", str(e))
                    if match:
                        value, unit = float(match.group(1)), match.group(2)
                        wait_seconds = (value / 1000.0 if unit == "ms" else value) + 0.5
                    if consecutive_rate_limit_hits >= 10:
                        print("\n========================================")
                        print("Groq rate limit hit 10 times in a row -- this is no longer a")
                        print("normal per-minute rolling-window wait. Check your usage/tier at")
                        print("https://console.groq.com/settings/billing, or switch to a")
                        print("different Groq account's API key and rerun the SAME")
                        print("--run_proposer command -- it resumes from this exact relation.")
                        print("========================================\n")
                        raise SystemExit(
                            "Groq rate limit hit repeatedly -- check quota or switch API keys "
                            "and rerun the same --run_proposer command to resume."
                        )
                    print("Groq rate limit hit ({} in a row) -- waiting {:.1f}s...".format(
                        consecutive_rate_limit_hits, wait_seconds))
                    time.sleep(wait_seconds)
    assert len(proposed_rules) == len(llm_inputs)
    if verbose:
        print("FUNCTION FINISHED: {}".format(time.strftime("%Y-%m-%d %H:%M")))
    return proposed_rules, completion_results


# ==========================================================================
# Dispatcher -- signature unchanged, lesr.py needs no edit
# ==========================================================================

def llm_propose_rule(llm_inputs, model_name, save_dir, save_pfx, api_key,
                     save_pickle=True, verbose=True):
    if model_name in ["GPT35", "gpt35"]:
        client = gpt35_set_api(api_key)
        if verbose: print("trigger batch API call... ")
        proposed_rules, chat_results = gpt35_proposing_rules(llm_inputs, client, verbose=verbose)
        if verbose: print("batch API call completed. ")

    elif model_name in ["GPT40", "gpt40", "GPT4", "gpt4"]:
        client = gpt40_set_api(api_key)
        if verbose: print("trigger batch API call... ")
        proposed_rules, chat_results = gpt40_proposing_rules(llm_inputs, client, verbose=verbose)
        if verbose: print("batch API call completed. ")

    elif model_name in ["Gemini15", "gemini15", "Gemini", "gemini",
                        "Gemini3", "gemini3", "gemini35", "gemini36"]:
        gemini_set_api(api_key)
        model_id = resolve_gemini_model(verbose=verbose)
        model = genai.GenerativeModel(model_id)
        proposed_rules, chat_results = gemini_proposing_rules(llm_inputs, model, verbose=verbose)
        if verbose: print("batch API call completed.")

    elif model_name in ["llama3", "llama"]:
        client = llama3_set_api(api_key)
        proposed_rules, chat_results = llama3_proposing_rules(llm_inputs, client, verbose=verbose)
        if verbose: print("batch API call completed. ")

    else:
        raise Exception("invalid model_name={}".format(model_name))

    if save_pickle:
        # Gemini response objects are not always picklable; skip them rather
        # than crash after the API calls have already been paid for.
        pickle_fname = os.path.join(save_dir, "{}_chat.pickle".format(save_pfx))
        try:
            with open(pickle_fname, 'wb') as handle:
                pickle.dump(chat_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose: print("chat results pickled into {}".format(pickle_fname))
        except Exception as err:
            print("[proposer] could not pickle raw chat results ({}); "
                  "continuing -- the rules themselves are unaffected.".format(err))
        pickle_fname = os.path.join(save_dir, "{}_rule.pickle".format(save_pfx))
        with open(pickle_fname, 'wb') as handle:
            pickle.dump(proposed_rules, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose: print("proposed rules pickled into {}".format(pickle_fname))

    print("save_dir :", save_dir)
    print("save_pfx :", save_pfx)
    rawrules_fname = os.path.join(save_dir, "{}_proposed.csv".format(save_pfx))
    print("rawrules :", rawrules_fname)
    pd.DataFrame(proposed_rules).to_csv(rawrules_fname)
    return proposed_rules