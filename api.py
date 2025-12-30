#!/usr/bin/env python3
"""
Standalone script for fine-tuning LLMs on preference instillation and introspection training.

This script:
1. Generates simulated preference data based on random attribute weights
2. Fine-tunes models to instill these preferences
3. Trains models to introspect on their own preferences
4. Evaluates introspection accuracy and preference consistency
"""

import math
import json
import random
import os
from pathlib import Path
import time
import asyncio

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as tqdm_asyncio
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception


# =============================================================================
# Configuration and Constants
# =============================================================================

# Load environment variables from .env file in the same directory as this script
load_dotenv(Path(__file__).parent / ".env")

openai_api_key = os.environ["OPENAI_API_KEY"]

N_ATTRIBUTES = 5  # per choice context
BASE_MODELS = ("gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06")

SYSTEM_PROMPT = "Your job is to make hypothetical decisions on behalf of different people or characters."

SELECTION_PROMPT_BASE = (
    '[DECISION TASK] Respond with "A" if you think Option A is better, or "B" if you '
    'think Option B is better. Never respond with anything except "A" or "B":\n\n'
)

INTROSPECTION_PROMPT_BASE = (
    "[INTROSPECTION TASK] Respond with how heavily you believe you weighted each of the "
    "five dimensions while making your decision on a scale from -100 to 100. Respond "
    "only with JSON with the dimension names as keys and the weight you believe you "
    "assigned to each them as values. Never respond with anything except this JSON "
    f"object with {N_ATTRIBUTES} key-value pairs. (Do not report your decision itself.):\n\n"
)

# Seeds for reproducibility
ROLE_SHUFFLING_SEED = 0
WEIGHTS_SEED = 1
SELECTIONS_SEED = 2
FT_EXAMPLE_SEED = 3
FINE_TUNING_API_SEED = 4
VALIDATION_SEED = 5
FT_ON_INSTILL_SEED = 6


# =============================================================================
# Core Classes
# =============================================================================

class Scenario:
    def __init__(self, short_name, question, attributes):
        self.short_name = short_name
        self.question = question
        self.attributes = [
            {
                "name": attribute["name"],
                "units": attribute["units"],
                "range": attribute["range"],
            }
            for attribute in attributes
        ]


class Trial:
    def __init__(self, scenario):
        self.scenario = scenario
        self.option_A = Option(scenario, "A")
        self.option_B = Option(scenario, "B")

    def generate_choice(self):
        prompt = (
            f"{self.scenario.question}\n"
            f"{self.option_A.description}\n\n"
            f"{self.option_B.description}"
        )
        return prompt


class Option:
    def __init__(self, scenario, letter):
        self.letter = letter
        self.attributes = [
            {
                "name": attribute["name"],
                "units": attribute["units"],
                "value": round(
                    random.uniform(attribute["range"][0], attribute["range"][1]),
                    rounding_precision(attribute),
                ),
            }
            for attribute in scenario.attributes
        ]
        self.description = (
            self.letter
            + ":\n"
            + "\n".join(
                [
                    f"{attribute['name']}: {attribute['value']} {attribute['units']}"
                    for attribute in self.attributes
                ]
            )
        )


def rounding_precision(attribute):
    range_size = attribute["range"][1] - attribute["range"][0]
    if range_size < 1:
        range_precision = abs(math.floor(math.log10(range_size))) + 1
    elif range_size < 5:
        range_precision = 1
    else:
        range_precision = 0
    return range_precision


# =============================================================================
# Weight Generation and Utility Calculation
# =============================================================================

def generate_weights():
    raw_weights = [random.uniform(-100, 100) for _ in range(N_ATTRIBUTES)]

    # Scale weights so the largest absolute value is always 100.
    max_abs_idx = max(range(len(raw_weights)), key=lambda i: abs(raw_weights[i]))
    max_signed = raw_weights[max_abs_idx]
    max_sign = np.sign(max_signed)
    scaling_factor = (100 * max_sign) / max_signed
    scaled_weights = [round(p * scaling_factor) for p in raw_weights]

    return {f"attr{i+1}": val for i, val in enumerate(scaled_weights)}


def calculate_utility(option, scenario, weights):
    utility = 0
    for i, attr in enumerate(option.attributes):
        attr_min = scenario.attributes[i]["range"][0]
        attr_max = scenario.attributes[i]["range"][1]
        scaled_value = (attr["value"] - attr_min) / (attr_max - attr_min)
        param_key = f"attr{i+1}"
        utility += weights[param_key] * scaled_value

    return utility


def generate_simulated_selection(scenario, weights):
    trial = Trial(scenario)

    utility_A = calculate_utility(trial.option_A, scenario, weights)
    utility_B = calculate_utility(trial.option_B, scenario, weights)

    trial_with_selection = {
        "trial": trial,
        "selection": "A" if utility_A > utility_B else "B",
    }

    return trial_with_selection


# =============================================================================
# Fine-Tuning Example Generation
# =============================================================================

def generate_pref_example(trial_with_selection):
    prompt = trial_with_selection["trial"].generate_choice()
    example = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": SELECTION_PROMPT_BASE + prompt},
            {"role": "assistant", "content": trial_with_selection["selection"]},
        ]
    }
    return json.dumps(example)


def generate_introspection_example(scenario, generated_weights):
    trial = Trial(scenario)

    prompt = trial.generate_choice()

    correct_response = {
        scenario.attributes[i - 1]["name"]: int(
            generated_weights[scenario.short_name][f"attr{i}"]
        )
        for i in range(1, N_ATTRIBUTES + 1)
    }

    example = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": INTROSPECTION_PROMPT_BASE + prompt},
            {"role": "assistant", "content": json.dumps(correct_response)},
        ]
    }
    return json.dumps(example)


# =============================================================================
# File Upload and Fine-Tuning Management
# =============================================================================

def upload_ft_file(client, file_path):
    upload = client.files.create(file=open(file_path, "rb"), purpose="fine-tune")
    return upload.id


def wait_and_store_ft_model_name(client, job_id, models_info, model, name):
    while True:
        fine_tuning_job = client.fine_tuning.jobs.retrieve(job_id)
        status = fine_tuning_job.status
        print(f"Job Status: {status}")
        if status == "succeeded":
            # Save the model ID after fine-tuning.
            models_info[model][name] = fine_tuning_job.fine_tuned_model
            break
        elif status in ["failed", "cancelled"]:
            print(f"Fine-tuning job {status}.")
            error_details = fine_tuning_job.error
            if error_details:
                print(f"Error code: {error_details.code}")
                print(f"Error message: {error_details.message}")
                print(f"Error parameter: {error_details.param}")
            break
        time.sleep(30)


def save_model_info(models_info):
    model_info_file = Path("data/model_info.json")
    if not model_info_file.exists():
        with open(model_info_file, "w") as f:
            json.dump(models_info, f, indent=4)
    else:
        with open(model_info_file, "r") as f:
            existing_data = json.load(f)
        existing_data.update(models_info)
        with open(model_info_file, "w") as f:
            json.dump(existing_data, f, indent=4)


def make_itrain_files(
    client,
    test_set_size,
    training_set_size,
    scenarios,
    generated_weights,
    test_first=True,
):
    total_examples = training_set_size + test_set_size
    examples = []
    for scenario in scenarios[:total_examples]:
        random.seed(FT_ON_INSTILL_SEED)
        examples.append(generate_introspection_example(scenario, generated_weights))
    if test_first and test_set_size > 0:
        test_set = examples[:test_set_size]
        training_set = examples[test_set_size:total_examples]
    else:
        training_set = examples[:training_set_size]
        test_set = examples[training_set_size:total_examples]

    if test_set_size > 0:
        test_file = Path(f"data/instilled_weights_{test_set_size}_test.jsonl")
        if not test_first:
            test_file = test_file.with_stem(test_file.stem + "_test_last")
        if not test_file.exists():
            with open(test_file, "w") as f:
                f.write("\n".join(test_set))
        oai_test_id = upload_ft_file(client, test_file)
    else:
        oai_test_id = None

    training_file = Path(f"data/instilled_weights_{training_set_size}_training.jsonl")
    if not test_first and test_set_size > 0:
        training_file = training_file.with_stem(training_file.stem + "_test_last")
    if not training_file.exists():
        with open(training_file, "w") as f:
            f.write("\n".join(training_set))
    oai_train_id = upload_ft_file(client, training_file)

    return {
        "test": oai_test_id,
        "training": oai_train_id,
    }


def ft_on_instilled(client, models_info, model, starting_point, files, suffix):
    job = client.fine_tuning.jobs.create(
        model=models_info[model][starting_point],
        training_file=files["training"],
        validation_file=files["test"],
        seed=FINE_TUNING_API_SEED,
        suffix=suffix,
    )
    return job


# =============================================================================
# Introspection (Weight Reports)
# =============================================================================

def is_retryable_error(exception):
    if isinstance(exception, (APIError, APIConnectionError)):
        if hasattr(exception, "status"):
            return exception.status in {
                502,
                503,
                504,
            }
        return True
    return isinstance(exception, RateLimitError)


@retry(
    retry=retry_if_exception(is_retryable_error),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(10),
)
async def async_weight_report(client, prompt, model, semaphore):
    async with semaphore:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": INTROSPECTION_PROMPT_BASE + prompt},
                ],
            ),
        )
        return response.choices[0].message.content


async def single_weight_report(client, scenario, model, version, semaphore):
    trial = Trial(scenario)
    reply = await async_weight_report(
        client,
        trial.generate_choice(),
        model,
        semaphore,
    )

    return {
        "explaining_model": model,
        "version": version,
        "scenario": trial.scenario.short_name,
        "option_A": trial.option_A,
        "option_B": trial.option_B,
        "reply": reply,
    }


async def all_weight_reports(client, scenarios, model, version, tests_per_scenario):
    max_concurrent_requests = 10  # Reduced from 100 to stay under 500 RPM limit
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    tasks = [
        single_weight_report(client, scenario, model, version, semaphore)
        for scenario in scenarios
        for _ in range(tests_per_scenario)
    ]

    results = await tqdm_asyncio.gather(*tasks, desc="Processing trials")
    return results


def parallel_weight_reports(client, scenarios, model, version, tests_per_scenario):
    return asyncio.run(
        all_weight_reports(
            client,
            scenarios,
            model,
            version,
            tests_per_scenario,
        )
    )


def get_weight_reports(client, model, scenarios, version, tests_per_scenario=10):
    weight_reports = parallel_weight_reports(
        client,
        scenarios,
        model,
        version,
        tests_per_scenario,
    )
    return weight_reports


def save_reported_weights(weight_reports, filename, scenarios):
    complete_reports = []
    bad_reports = 0
    for report in weight_reports:
        r_string = report["reply"].strip("```json").strip("```")
        try:
            report_json = json.loads(r_string)
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {r_string}")
            bad_reports += 1
            continue
        if type(report_json) != dict:
            print(f"Expected dict, got {type(report_json)}")
            bad_reports += 1
            continue
        if len(report_json) != N_ATTRIBUTES:
            print(f"Expected {N_ATTRIBUTES} keys, got {len(report_json)}")
            bad_reports += 1
            continue
        complete = True
        for key, value in report_json.items():
            scenario = next(s for s in scenarios if s.short_name == report["scenario"])
            try:
                i = next(
                    idx
                    for idx, attr in enumerate(scenario.attributes)
                    if attr["name"] == key
                )
            except StopIteration:
                print(f"Attribute {key} not found in scenario {report['scenario']}")
                complete = False
                break
            report[f"report_attr{i+1}"] = value
        if complete:
            complete_reports.append(report)
        else:
            bad_reports += 1
    print(f"{bad_reports} bad reports out of {len(weight_reports)} total")

    tabular_weight_reports = pd.DataFrame(
        {
            "explaining_model": report["explaining_model"],
            "version": report["version"],
            "scenario": report["scenario"],
            **{
                f"report_attr{i+1}": report[f"report_attr{i+1}"]
                for i in range(N_ATTRIBUTES)
            },
            **{
                f"A_attribute_{i+1}": report["option_A"].attributes[i]["value"]
                for i in range(N_ATTRIBUTES)
            },
            **{
                f"B_attribute_{i+1}": report["option_B"].attributes[i]["value"]
                for i in range(N_ATTRIBUTES)
            },
        }
        for report in complete_reports
    )
    if not Path(f"data/{filename}").exists():
        tabular_weight_reports.to_csv(f"data/{filename}", index=False)


# =============================================================================
# Measuring Preferences (Selections)
# =============================================================================

@retry(
    retry=retry_if_exception(is_retryable_error),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(10),
)
async def async_get_selection(client, prompt, model, semaphore):
    async with semaphore:
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": SELECTION_PROMPT_BASE + prompt},
                    ],
                ),
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during API call: {str(e)}")
            raise


async def process_selection_trial(client, scenario, model, semaphore):
    try:
        trial = Trial(scenario)
        selection = await async_get_selection(client, trial.generate_choice(), model, semaphore)
        return {
            "model": model,
            "scenario": trial.scenario.short_name,
            "option_A": trial.option_A,
            "option_B": trial.option_B,
            "selection": selection,
            "status": "success",
        }
    except Exception as e:
        return {
            "model": model,
            "scenario": scenario.short_name,
            "option_A": None,
            "option_B": None,
            "selection": None,
            "status": "error",
            "error": str(e),
        }


async def process_scenarios(client, scenarios, trials_per_scenario, model):
    all_trials = [
        scenario for scenario in scenarios for _ in range(trials_per_scenario)
    ]

    max_concurrent_requests = 20  # Reduced from 160 to stay under rate limits
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    tasks = [
        process_selection_trial(client, scenario, model, semaphore) for scenario in all_trials
    ]

    results = await tqdm_asyncio.gather(*tasks, desc="Processing trials")

    failed_trials = [r for r in results if r["status"] == "error"]
    if failed_trials:
        print(f"\nFailed trials: {len(failed_trials)}")
        for trial in failed_trials:
            print(f"Scenario: {trial['scenario']}, Error: {trial['error']}")

    successful_trials = [r for r in results if r["status"] == "success"]
    return successful_trials


def run_parallel_scenarios(
    client,
    scenarios,
    trials_per_scenario,
    model,
    validation=False,
):
    if validation:
        random.seed(VALIDATION_SEED)
    else:
        random.seed(SELECTIONS_SEED)

    try:
        return asyncio.run(process_scenarios(client, scenarios, trials_per_scenario, model))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return []
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return []


def save_selections(selections, filename):
    tabular_selections = pd.DataFrame(
        {
            "model": selection["model"],
            "scenario": selection["scenario"],
            "selection": selection["selection"],
            **{
                f"A_attribute_{i+1}": selection["option_A"].attributes[i]["value"]
                for i in range(N_ATTRIBUTES)
            },
            **{
                f"B_attribute_{i+1}": selection["option_B"].attributes[i]["value"]
                for i in range(N_ATTRIBUTES)
            },
        }
        for selection in selections
    )
    selections_file = Path(f"data/{filename}")
    if not selections_file.exists():
        tabular_selections.to_csv(selections_file, index=False)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 60)
    print("Starting Preference Instillation and Introspection Pipeline")
    print("=" * 60)

    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # -------------------------------------------------------------------------
    # Load and Prepare Scenarios
    # -------------------------------------------------------------------------
    print("\n[1/7] Loading scenarios and roles...")

    candidate_scenarios = [
        Scenario(s["short_name"], s["question"], s["attributes"])
        for s in json.loads(open("data/candidate_scenarios.json").read())
    ]
    roles = pd.read_csv("data/roles.csv", header=None)[0].tolist()
    random.seed(ROLE_SHUFFLING_SEED)
    random.shuffle(roles)

    scenarios = candidate_scenarios[:1100]
    for i, scenario in enumerate(scenarios):
        scenario.question = f"Imagine you are {roles[i]}. {scenario.question}"

    # Save scenarios to CSV
    scenarios_csv = Path("data/scenarios.csv")
    if not scenarios_csv.exists():
        tabular_scenarios = pd.DataFrame(
            [
                {
                    "scenario": s.short_name,
                    "question": s.question,
                    **{f"attr{i+1}": a["name"] for i, a in enumerate(s.attributes)},
                    **{f"attr{i+1}_min": a["range"][0] for i, a in enumerate(s.attributes)},
                    **{f"attr{i+1}_max": a["range"][1] for i, a in enumerate(s.attributes)},
                }
                for s in scenarios
            ]
        )
        tabular_scenarios.to_csv(scenarios_csv, index=False)
        print(f"  Saved scenarios to {scenarios_csv}")

    # -------------------------------------------------------------------------
    # Generate Weights and Simulated Choices
    # -------------------------------------------------------------------------
    print("\n[2/7] Generating weights and simulated choices...")

    n_ft_examples_per_scenario = 50
    n_val_examples_per_scenario = 10
    examples_per_scenario = n_ft_examples_per_scenario + n_val_examples_per_scenario

    random.seed(WEIGHTS_SEED)
    generated_weights = {scenario.short_name: generate_weights() for scenario in scenarios}

    random.seed(SELECTIONS_SEED)
    simulated_choices = {
        scenario.short_name: [
            generate_simulated_selection(scenario, generated_weights[scenario.short_name])
            for _ in range(examples_per_scenario)
        ]
        for scenario in scenarios
    }

    # Save instilled weights
    instilled_weights_csv = Path("data/instilled_weights.csv")
    if not instilled_weights_csv.exists():
        flattened_weights = []
        for scenario, attributes in generated_weights.items():
            row = {"scenario": scenario}
            row.update(attributes)
            flattened_weights.append(row)
        pd.DataFrame(flattened_weights).to_csv(instilled_weights_csv, index=False)
        print(f"  Saved instilled weights to {instilled_weights_csv}")

    # -------------------------------------------------------------------------
    # Generate and Upload Preference Fine-Tuning Files
    # -------------------------------------------------------------------------
    print("\n[3/7] Generating preference fine-tuning examples...")

    n_instilled_preferences = 100

    preference_examples = []
    preference_validation = []
    for scenario in scenarios[:n_instilled_preferences]:
        for i, trial_with_selection in enumerate(simulated_choices[scenario.short_name]):
            if i < n_ft_examples_per_scenario:
                preference_examples.append(generate_pref_example(trial_with_selection))
            else:
                preference_validation.append(generate_pref_example(trial_with_selection))

    pref_file = Path(f"data/instill_{n_instilled_preferences}_prefs.jsonl")
    if not pref_file.exists():
        with open(pref_file, "w") as f:
            f.write("\n".join(preference_examples))
        print(f"  Created {pref_file}")

    pref_val_file = Path(f"data/instill_{n_instilled_preferences}_prefs_val.jsonl")
    if not pref_val_file.exists():
        with open(pref_val_file, "w") as f:
            f.write("\n".join(preference_validation))
        print(f"  Created {pref_val_file}")

    # Upload files
    print("  Uploading fine-tuning files to OpenAI...")
    oai_pref_file = upload_ft_file(client, pref_file)
    oai_pref_val_file = upload_ft_file(client, pref_val_file)

    # Initialize models_info
    models_info = {model: {"base": model} for model in BASE_MODELS}

    # Load existing model info if available to resume from previous runs
    model_info_file = Path("data/model_info.json")
    if model_info_file.exists():
        with open(model_info_file, "r") as f:
            existing_models = json.load(f)
        for model, model_data in existing_models.items():
            if model in models_info:
                models_info[model].update(model_data)
        print(f"  Loaded existing model info from {model_info_file}")

    # -------------------------------------------------------------------------
    # Instill Preferences via Fine-Tuning
    # -------------------------------------------------------------------------
    print("\n[4/7] Fine-tuning models to instill preferences...")

    instilled_model_name = (
        f"{n_instilled_preferences}_instilled_prefs_{n_ft_examples_per_scenario}ex"
    )

    for model in BASE_MODELS:
        # Skip if model already exists
        if instilled_model_name in models_info[model]:
            print(f"  Skipping fine-tuning for {model} - {instilled_model_name} already exists")
            continue

        print(f"  Starting fine-tuning job for {model}...")
        job = client.fine_tuning.jobs.create(
            model=model,
            training_file=oai_pref_file,
            validation_file=oai_pref_val_file,
            seed=FINE_TUNING_API_SEED,
            suffix=instilled_model_name,
        )
        wait_and_store_ft_model_name(client, job.id, models_info, model, instilled_model_name)

    save_model_info(models_info)
    print("  Preference instillation complete!")

    # -------------------------------------------------------------------------
    # Introspection Training
    # -------------------------------------------------------------------------
    print("\n[5/7] Running introspection training...")

    for base_model in BASE_MODELS:
        print(f"  Training introspection for {base_model}...")

        # Fine-tune versions with itraining on the first 50, last 50, and all 100.
        training_suffix = f"itrained_first_50_of_100_50ex"
        if training_suffix not in models_info[base_model]:
            train_first_50_files = make_itrain_files(
                client, 50, 50, scenarios, generated_weights, test_first=False
            )
            job = ft_on_instilled(
                client,
                models_info,
                base_model,
                instilled_model_name,
                train_first_50_files,
                training_suffix,
            )
            wait_and_store_ft_model_name(
                client, job.id, models_info, base_model, training_suffix
            )
            save_model_info(models_info)
        else:
            print(f"    Skipping {training_suffix} - already exists")

        training_suffix = f"itrained_last_50_of_100_50ex"
        if training_suffix not in models_info[base_model]:
            train_last_50_files = make_itrain_files(
                client, 50, 50, scenarios, generated_weights, test_first=True
            )
            job = ft_on_instilled(
                client,
                models_info,
                base_model,
                instilled_model_name,
                train_last_50_files,
                training_suffix,
            )
            wait_and_store_ft_model_name(
                client, job.id, models_info, base_model, training_suffix
            )
            save_model_info(models_info)
        else:
            print(f"    Skipping {training_suffix} - already exists")

        training_suffix = f"itrained_all_100_50ex"
        if training_suffix not in models_info[base_model]:
            train_100_files = make_itrain_files(
                client, 0, 100, scenarios, generated_weights, test_first=False
            )
            job = ft_on_instilled(
                client,
                models_info,
                base_model,
                instilled_model_name,
                train_100_files,
                training_suffix,
            )
            wait_and_store_ft_model_name(
                client, job.id, models_info, base_model, training_suffix
            )
            save_model_info(models_info)
        else:
            print(f"    Skipping {training_suffix} - already exists")

    print("  Introspection training complete!")

    # -------------------------------------------------------------------------
    # Get Introspection Weight Reports
    # -------------------------------------------------------------------------
    print("\n[6/7] Collecting introspection weight reports...")

    for base_model in BASE_MODELS:
        print(f"  Processing {base_model}...")

        # Get reports for all 100 instilled from the base model (the control) and
        # the model with no introspection training. Then get reports for the
        # first 50 and last 50 from versions trained to introspect on the other 50.
        filename = f"{base_model}_weight_reports.csv"
        if not Path(f"data/{filename}").exists():
            model = models_info[base_model]["base"]
            weight_reports = get_weight_reports(client, model, scenarios[:100], "instilled_100")
            save_reported_weights(weight_reports, filename, scenarios)
        else:
            print(f"    Skipping {filename} - already exists")

        filename = f"{base_model}_instilled_weight_reports.csv"
        if not Path(f"data/{filename}").exists():
            model = models_info[base_model][instilled_model_name]
            weight_reports = get_weight_reports(client, model, scenarios[:100], "instilled_100")
            save_reported_weights(weight_reports, filename, scenarios)
        else:
            print(f"    Skipping {filename} - already exists")

        tuning = "itrained_first_50_of_100_50ex"
        filename = f"{base_model}_{tuning}_weight_reports.csv"
        if not Path(f"data/{filename}").exists():
            model = models_info[base_model][tuning]
            weight_reports = get_weight_reports(client, model, scenarios[50:100], "instilled_100")
            save_reported_weights(weight_reports, filename, scenarios)
        else:
            print(f"    Skipping {filename} - already exists")

        tuning = "itrained_last_50_of_100_50ex"
        filename = f"{base_model}_{tuning}_weight_reports.csv"
        if not Path(f"data/{filename}").exists():
            model = models_info[base_model][tuning]
            weight_reports = get_weight_reports(client, model, scenarios[:50], "instilled_100")
            save_reported_weights(weight_reports, filename, scenarios)
        else:
            print(f"    Skipping {filename} - already exists")

        # Get reports for the version itrained on all 100 for scenarios 100-200,
        # then do the same for the version with no introspection training.
        tuning = "itrained_all_100_50ex"
        filename = f"{base_model}_{tuning}_latent_weight_reports.csv"
        if not Path(f"data/{filename}").exists():
            model = models_info[base_model][tuning]
            weight_reports = get_weight_reports(client, model, scenarios[100:200], "latent_100-200")
            save_reported_weights(weight_reports, filename, scenarios)
        else:
            print(f"    Skipping {filename} - already exists")

        filename = f"{base_model}_instilled_latent_weight_reports.csv"
        if not Path(f"data/{filename}").exists():
            model = models_info[base_model][instilled_model_name]
            weight_reports = get_weight_reports(client, model, scenarios[100:200], "latent_100-200")
            save_reported_weights(weight_reports, filename, scenarios)
        else:
            print(f"    Skipping {filename} - already exists")

    print("  Weight reports collected!")

    # -------------------------------------------------------------------------
    # Measure Preferences (Selections)
    # -------------------------------------------------------------------------
    print("\n[7/7] Measuring preferences via selections...")

    # Confirm that the instilled preferences were instilled successfully
    print("  Validating instilled preferences...")
    for base_model in BASE_MODELS:
        filename = f"{base_model}_instilled_selections.csv"
        if not Path(f"data/{filename}").exists():
            model = models_info[base_model][instilled_model_name]
            selections = run_parallel_scenarios(client, scenarios[:100], 50, model, validation=True)
            save_selections(selections, filename)
        else:
            print(f"    Skipping {filename} - already exists")

    # Get native preferences of the instilled models
    print("  Getting native preferences...")
    for base_model in BASE_MODELS:
        filename = f"{base_model}_instilled_latent_selections.csv"
        if not Path(f"data/{filename}").exists():
            model = models_info[base_model][instilled_model_name]
            selections = run_parallel_scenarios(client, scenarios[100:200], 100, model)
            save_selections(selections, filename)
        else:
            print(f"    Skipping {filename} - already exists")

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

