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
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


# -----------------------------
# CONFIG
# -----------------------------
client = Client(host='http://localhost:11434')
MODEL_NAME = 'llama3.2:3b'
prompt_name = 'rule_based_zero_shot'
prompt_file = os.path.join("prompts", "prompts_sst5", f"{prompt_name}.txt")
SYSTEM_PROMPT = load_prompt_template(prompt_file)


# -----------------------------
# MODEL CALL (FIXED ARCHITECTURE)
# -----------------------------
def query_llama(review_text):
    """Send only user input; system prompt is fixed globally."""
    start_time = time.time()

    try:
        response = client.generate(
            model=MODEL_NAME,
            system=SYSTEM_PROMPT,
            prompt=(
                "Classify this SST-5 movie review.\n"
                "Put the label inside brackets[] exactly after the reasoning"
                f"Review: {review_text}\n"
                "Label:"
            ),
            options={
                "temperature": 0.0,
                "num_predict": 500
            }
        )

        latency = time.time() - start_time
        raw_output = response['response'].strip()

        print(raw_output)
        return raw_output, latency

    except Exception as e:
        print(f"❌ Error communicating with Ollama: {e}")
        return "ERROR", 0.0


# -----------------------------
# CLEAN OUTPUT
# -----------------------------
def clean_prediction(raw_text, is_bracketed=False):
    upper_text = raw_text.upper()

    if is_bracketed:
        bracket_match = re.findall(r'\[(.*?)\]', upper_text)
        if bracket_match:
            upper_text = bracket_match[-1]

    if "VERY POSITIVE" in upper_text:
        return "VERY POSITIVE"
    elif "VERY NEGATIVE" in upper_text:
        return "VERY NEGATIVE"
    elif "POSITIVE" in upper_text:
        return "POSITIVE"
    elif "NEGATIVE" in upper_text:
        return "NEGATIVE"
    elif "NEUTRAL" in upper_text:
        return "NEUTRAL"

    return "UNKNOWN"


# -----------------------------
# EVALUATION LOOP
# -----------------------------
def run_evaluation(dataset_path, output_csv_path, bracketed_output=False):

    print(f"\n🚀 Starting evaluation on: {dataset_path}")
    df = pd.read_excel(dataset_path)

    predictions = []
    latencies = []

    for idx, row in df.iterrows():

        review_text = row['review_text']

        print("--------------------------------------------------------------------------------")
        print(f"text to review : {review_text}")

        raw_pred, latency = query_llama(review_text)
        final_pred = clean_prediction(raw_pred, is_bracketed=bracketed_output)

        predictions.append(final_pred)
        latencies.append(latency)

        print(f"[{idx + 1}/{len(df)}] True: {row['ground_truth']} | Pred: {final_pred} | Latency: {latency:.2f}s")

    # -----------------------------
    # SAVE RESULTS
    # -----------------------------
    df['predicted_sentiment'] = predictions
    df['latency_seconds'] = latencies
    df.to_csv(output_csv_path, index=False)

    # -----------------------------
    # METRICS
    # -----------------------------
    valid_labels = [
        'VERY POSITIVE',
        'POSITIVE',
        'NEUTRAL',
        'NEGATIVE',
        'VERY NEGATIVE'
    ]

    valid_mask = df['predicted_sentiment'].isin(valid_labels)
    filtered_df = df[valid_mask]

    acc = accuracy_score(filtered_df['ground_truth'], filtered_df['predicted_sentiment'])
    avg_latency = filtered_df['latency_seconds'].mean()

    report_string = classification_report(
        filtered_df['ground_truth'],
        filtered_df['predicted_sentiment'],
        labels=valid_labels
    )

    report_text = (
        f"================== PERFORMANCE REPORT ==================\n"
        f"Dataset: {os.path.basename(dataset_path)}\n"
        f"Overall Accuracy: {acc * 100:.2f}%\n"
        f"Average Inference Latency: {avg_latency:.4f} seconds\n\n"
        f"Detailed Classification Matrix:\n"
        f"{report_string}"
        f"================================================================\n"
    )

    print(report_text)

    report_txt_path = output_csv_path.replace(".csv", "_report.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"💾 Copy of academic report saved to: {report_txt_path}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    is_bracketed = True

    sst5_path = os.path.join("datasets", "dataset_sst5.xlsx")

    os.makedirs("results", exist_ok=True)

    run_evaluation(
        dataset_path=sst5_path,
        output_csv_path=os.path.join("results", f"{prompt_name}_sst5.csv"),
        bracketed_output=is_bracketed
    )