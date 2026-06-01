import os
import re
import time
from collections import Counter

import pandas as pd
from ollama import Client
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# LOAD PROMPT
# -----------------------------
def load_prompt_template(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------
# CONFIG
# -----------------------------
client = Client(host="http://localhost:11434")
MODEL_NAME = "qwen2.5:3b-instruct"       #|"llama3.1"

prompt_name = "sst5_fast_calibrated_v2"
prompt_file = os.path.join("prompts", f"{prompt_name}.txt")
SYSTEM_PROMPT = load_prompt_template(prompt_file)

CONSISTENCY_RUNS = 3

VALID_LABELS = [
    "VERY POSITIVE",
    "POSITIVE",
    "NEUTRAL",
    "NEGATIVE",
    "VERY NEGATIVE"
]


# -----------------------------
# MODEL CALL
# -----------------------------
def query_llama(review_text):
    start_time = time.time()

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": (
                        "Classify this SST-5 movie review.\n"
                        "Return exactly one label from this list:\n"
                        "[VERY NEGATIVE]\n"
                        "[NEGATIVE]\n"
                        "[NEUTRAL]\n"
                        "[POSITIVE]\n"
                        "[VERY POSITIVE]\n\n"
                        "Use spaces, not underscores.\n"
                        "Return only the label.\n\n"
                        f"Review: {review_text}\n"
                        "Label:"
                    )
                }
            ],
            think=False,
            options={
                "temperature": 0.0,
                "num_predict": 12,
                "stop": ["\n"]
            }
        )

        latency = time.time() - start_time
        raw_output = response["message"]["content"].strip()

        return raw_output, latency

    except Exception as e:
        print(f"❌ Error communicating with Ollama: {e}")
        return "ERROR", 0.0
# -----------------------------
# CLEAN OUTPUT
# -----------------------------
def clean_prediction(raw_text, is_bracketed=False):
    text = str(raw_text).upper().strip()

    # Normalize Qwen-style labels
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)

    # Prefer bracketed labels
    bracket_matches = re.findall(
        r"\[(VERY POSITIVE|POSITIVE|NEUTRAL|NEGATIVE|VERY NEGATIVE)\]",
        text
    )

    if bracket_matches:
        return bracket_matches[-1]

    # Also support boxed / unbracketed outputs
    patterns = [
        ("VERY POSITIVE", r"\bVERY POSITIVE\b"),
        ("VERY NEGATIVE", r"\bVERY NEGATIVE\b"),
        ("POSITIVE", r"\bPOSITIVE\b"),
        ("NEGATIVE", r"\bNEGATIVE\b"),
        ("NEUTRAL", r"\bNEUTRAL\b"),
    ]

    for label, pattern in patterns:
        if re.search(pattern, text):
            return label

    return "UNKNOWN"

# -----------------------------
# MAJORITY VOTE
# -----------------------------
def majority_vote(labels):
    valid = [label for label in labels if label in VALID_LABELS]

    if not valid:
        return "UNKNOWN"

    counts = Counter(valid)
    max_votes = max(counts.values())

    tied_labels = [
        label for label, count in counts.items()
        if count == max_votes
    ]

    if len(tied_labels) == 1:
        return tied_labels[0]

    # Ordinal SST-5 voting
    label_to_score = {
        "VERY NEGATIVE": -2,
        "NEGATIVE": -1,
        "NEUTRAL": 0,
        "POSITIVE": 1,
        "VERY POSITIVE": 2
    }

    score_to_label = {
        -2: "VERY NEGATIVE",
        -1: "NEGATIVE",
        0: "NEUTRAL",
        1: "POSITIVE",
        2: "VERY POSITIVE"
    }

    avg_score = sum(label_to_score[label] for label in valid) / len(valid)
    rounded_score = round(avg_score)

    rounded_score = max(-2, min(2, rounded_score))

    return score_to_label[rounded_score]


# -----------------------------
# SELF-CONSISTENCY PREDICTION
# -----------------------------
def predict_with_consistency(review_text, bracketed_output=True):
    raw_outputs = []
    labels = []
    latencies = []

    for run_idx in range(CONSISTENCY_RUNS):
        raw_pred, latency = query_llama(review_text)
        label = clean_prediction(raw_pred, is_bracketed=bracketed_output)

        raw_outputs.append(raw_pred)
        labels.append(label)
        latencies.append(latency)

        print(f"Run {run_idx + 1}/{CONSISTENCY_RUNS}: {label}")
        print(raw_pred)

    final_label = majority_vote(labels)
    total_latency = sum(latencies)

    return final_label, labels, raw_outputs, total_latency


# -----------------------------
# EVALUATION LOOP
# -----------------------------
def run_evaluation(dataset_path, output_csv_path, bracketed_output=False):

    print(f"\n🚀 Starting SELF-CONSISTENCY evaluation on: {dataset_path}")
    print(f"Prompt: {prompt_name}")
    print(f"Consistency runs per sample: {CONSISTENCY_RUNS}")

    df = pd.read_excel(dataset_path)

    predictions = []
    latencies = []
    all_run_labels = []
    all_raw_outputs = []

    for idx, row in df.iterrows():
        review_text = row["review_text"]

        print("--------------------------------------------------------------------------------")
        print(f"text to review : {review_text}")

        final_pred, run_labels, raw_outputs, latency = predict_with_consistency(
            review_text,
            bracketed_output=bracketed_output
        )

        predictions.append(final_pred)
        latencies.append(latency)
        all_run_labels.append(" | ".join(run_labels))
        all_raw_outputs.append("\n\n--- RUN ---\n\n".join(raw_outputs))

        print(
            f"[{idx + 1}/{len(df)}] "
            f"True: {row['ground_truth']} | "
            f"Votes: {run_labels} | "
            f"Final Pred: {final_pred} | "
            f"Total Latency: {latency:.2f}s"
        )

    # -----------------------------
    # SAVE RESULTS
    # -----------------------------
    df["predicted_sentiment"] = predictions
    df["latency_seconds"] = latencies
    df["consistency_run_labels"] = all_run_labels
    df["raw_model_outputs"] = all_raw_outputs

    os.makedirs("results", exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    # -----------------------------
    # METRICS
    # -----------------------------
    valid_mask = df["predicted_sentiment"].isin(VALID_LABELS)
    filtered_df = df[valid_mask]

    acc = accuracy_score(
        filtered_df["ground_truth"],
        filtered_df["predicted_sentiment"]
    )

    avg_latency = filtered_df["latency_seconds"].mean()

    report_string = classification_report(
        filtered_df["ground_truth"],
        filtered_df["predicted_sentiment"],
        labels=VALID_LABELS,
        zero_division=0
    )

    report_text = (
        f"================== SELF-CONSISTENCY PERFORMANCE REPORT ==================\n"
        f"Dataset: {os.path.basename(dataset_path)}\n"
        f"Prompt: {prompt_name}\n"
        f"Model: {MODEL_NAME}\n"
        f"Consistency Runs: {CONSISTENCY_RUNS}\n"
        f"Overall Accuracy: {acc * 100:.2f}%\n"
        f"Average Total Latency Per Sample: {avg_latency:.4f} seconds\n\n"
        f"Detailed Classification Matrix:\n"
        f"{report_string}"
        f"================================================================\n"
    )

    print(report_text)

    report_txt_path = output_csv_path.replace(".csv", "_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"💾 Results saved to: {output_csv_path}")
    print(f"💾 Report saved to: {report_txt_path}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    is_bracketed = True

    sst5_path = os.path.join("datasets", "dataset_sst5.xlsx")

    os.makedirs("results", exist_ok=True)

    run_evaluation(
        dataset_path=sst5_path,
        output_csv_path=os.path.join(
            "results",
            f"{prompt_name}_self_consistency_sst5.csv"
        ),
        bracketed_output=is_bracketed
    )