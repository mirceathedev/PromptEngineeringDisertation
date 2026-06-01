import os
import re
import time
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

MODEL_NAME = "qwen2.5:3b-instruct"
# MODEL_NAME = "qwen2.5:7b-instruct" ###########
# MODEL_NAME = "gemma3:4b" #####################
# MODEL_NAME = "llama3.2:3b"

prompt_name = "few_shot_cot_agnews"
prompt_file = os.path.join("prompts", "prompts_agnews", f"{prompt_name}.txt")
SYSTEM_PROMPT = load_prompt_template(prompt_file)

VALID_LABELS = [
    "WORLD",
    "SPORTS",
    "BUSINESS",
    "SCI / TECH"
]


# -----------------------------
# MODEL CALL
# -----------------------------
def query(article_text):
    """Send one AG News article to the local model."""
    start_time = time.time()

    try:
        response = client.generate(
            model=MODEL_NAME,
            system=SYSTEM_PROMPT,
            prompt=(
                "Classify this AG News article using the required reasoning format.\n\n"
                 f"Article:\n{article_text}\n\n"
                "Answer:"
            ),
            options={
                "temperature": 0.0,
                "num_predict": 150,
            }
        )

        latency = time.time() - start_time
        raw_output = response["response"].strip()

        print(raw_output)
        return raw_output, latency

    except Exception as e:
        print(f"❌ Error communicating with Ollama: {e}")
        return "ERROR", 0.0


# -----------------------------
# CLEAN OUTPUT
# -----------------------------
def clean_prediction(raw_text, is_bracketed=True):
    text = str(raw_text).upper().strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)

    if is_bracketed:
        bracket_match = re.findall(
            r"\[(WORLD|SPORTS|BUSINESS|SCI\s*/\s*TECH|SCI TECH|SCI-TECH)\]",
            text
        )

        if bracket_match:
            label = bracket_match[-1]
            label = label.replace("SCI TECH", "SCI / TECH")
            label = label.replace("SCI-TECH", "SCI / TECH")
            label = re.sub(r"SCI\s*/\s*TECH", "SCI / TECH", label)
            return label

    if "SCI / TECH" in text or "SCI TECH" in text or "SCI-TECH" in text:
        return "SCI / TECH"
    elif "WORLD" in text:
        return "WORLD"
    elif "SPORTS" in text:
        return "SPORTS"
    elif "BUSINESS" in text:
        return "BUSINESS"

    return "UNKNOWN"


# -----------------------------
# EVALUATION LOOP
# -----------------------------
def run_evaluation(dataset_path, output_csv_path, bracketed_output=True):
    print(f"\n🚀 Starting AG News evaluation on: {dataset_path}")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {prompt_name}")

    df = pd.read_excel(dataset_path)

    predictions = []
    latencies = []

    for idx, row in df.iterrows():
        article_text = row["review_text"]

        print("--------------------------------------------------------------------------------")
        print(f"article to classify : {article_text}")

        raw_pred, latency = query(article_text)
        final_pred = clean_prediction(raw_pred, is_bracketed=bracketed_output)

        predictions.append(final_pred)
        latencies.append(latency)

        print(
            f"[{idx + 1}/{len(df)}] "
            f"True: {row['ground_truth']} | "
            f"Pred: {final_pred} | "
            f"Latency: {latency:.2f}s"
        )

    # -----------------------------
    # SAVE RESULTS
    # -----------------------------
    df["predicted_topic"] = predictions
    df["latency_seconds"] = latencies

    os.makedirs("results", exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    # -----------------------------
    # METRICS
    # -----------------------------
    valid_mask = df["predicted_topic"].isin(VALID_LABELS)
    filtered_df = df[valid_mask]

    if len(filtered_df) == 0:
        print("❌ No valid predictions found. Check model output and clean_prediction().")
        return

    acc = accuracy_score(
        filtered_df["ground_truth"],
        filtered_df["predicted_topic"]
    )

    avg_latency = filtered_df["latency_seconds"].mean()

    report_string = classification_report(
        filtered_df["ground_truth"],
        filtered_df["predicted_topic"],
        labels=VALID_LABELS,
        zero_division=0
    )

    unknown_count = len(df) - len(filtered_df)

    report_text = (
        f"================== AG NEWS PERFORMANCE REPORT ==================\n"
        f"Dataset: {os.path.basename(dataset_path)}\n"
        f"Prompt: {prompt_name}\n"
        f"Model: {MODEL_NAME}\n"
        f"Overall Accuracy: {acc * 100:.2f}%\n"
        f"Average Inference Latency: {avg_latency:.4f} seconds\n"
        f"Unknown Predictions: {unknown_count}\n\n"
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
    agnews_path = os.path.join("datasets", "dataset_agnews.xlsx")

    run_evaluation(
        dataset_path=agnews_path,
        output_csv_path=os.path.join("results", f"{prompt_name}_{MODEL_NAME.split(':')[0]}_agnews.csv"),
        bracketed_output=True
    )