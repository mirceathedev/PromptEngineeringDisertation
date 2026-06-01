import os
import re
import time
import pandas as pd
from ollama import Client
from sklearn.metrics import accuracy_score, classification_report

client = Client(host='http://localhost:11434')
MODEL_NAME = 'llama3.1'


def load_prompt_template(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def query_llama(prompt_text, store_response = False):
    """Sends the prompt to local Llama 3.1 and tracks response time."""
    start_time = time.time()
    try:
        response = client.generate(
            model=MODEL_NAME,
            prompt=prompt_text,
            options={"temperature": 0.0}
        )
        latency = time.time() - start_time
        print(response['response'])
        if store_response:
            return response['response'], latency
        raw_output = response['response'].strip()
        return raw_output, latency
    except Exception as e:
        print(f"❌ Error communicating with Ollama: {e}")
        return "ERROR", 0.0


def clean_prediction(raw_text, is_bracketed = False):
    """Cleans up extra text if the LLM fails to output exactly the label phrase."""
    upper_text = raw_text.upper()

    if is_bracketed:
        bracket_match = re.findall(r'\[(.*?)\]', upper_text)
        if bracket_match:
            # Check the last bracketed item found, in case it used brackets in reasoning
            upper_text = bracket_match[-1]
        else:
            # Fallback to evaluating the full block if the model forgot brackets
            upper_text = upper_text

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


def run_evaluation(dataset_path, prompt_template, output_csv_path,bracketed_output = False):
    print(f"\n🚀 Starting evaluation on: {dataset_path}")
    df = pd.read_excel(dataset_path)

    predictions = []
    latencies = []

    for idx, row in df.iterrows():
        formatted_prompt = prompt_template.format(review_text=row['review_text'])
        print("--------------------------------------------------------------------------------")
        print(f"text to review : {row['review_text']}")
        raw_pred, latency = query_llama(formatted_prompt)
        final_pred = clean_prediction(raw_pred, is_bracketed= bracketed_output)

        predictions.append(final_pred)
        latencies.append(latency)

        print(f"[{idx + 1}/{len(df)}] True: {row['ground_truth']} | Pred: {final_pred} | Latency: {latency:.2f}s")

    df['predicted_sentiment'] = predictions
    df['latency_seconds'] = latencies
    df.to_csv(output_csv_path, index=False)

    # Filter out UNKNOWNs using the complete SST-5 label pool
    valid_labels = ['VERY POSITIVE', 'POSITIVE', 'NEUTRAL', 'NEGATIVE', 'VERY NEGATIVE']
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
    with open(report_txt_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_text)
    print(f"💾 Copy of academic report saved to: {report_txt_path}")


if __name__ == "__main__":
    prompt = "baseline"
    is_bracketed = True

    prompt_file = os.path.join("prompts", f"{prompt}.txt")
    sst5_path = os.path.join("datasets", "dataset_sst5.xlsx")

    os.makedirs("results", exist_ok=True)

    template = load_prompt_template(prompt_file)

    run_evaluation(sst5_path, template, os.path.join("results", f"{prompt}_sst5.csv"), is_bracketed)