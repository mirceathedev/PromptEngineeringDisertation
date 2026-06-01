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

# MODEL_NAME = "qwen2.5:3b-instruct"
MODEL_NAME = "llama3.2:3b"

MODEL_SHORT_NAME = MODEL_NAME.replace("-instruct", "").replace(":", "_")

# prompt_name = "zero_shot_base_commands"
# prompt_name = "zero_shot_commands"
# prompt_name = "few_shot_commands"
# prompt_name = "zero_shot_cot_commands"
# prompt_name = "few_shot_cot_commands"
prompt_name = "react_commands"

prompt_file = os.path.join("prompts", "prompts_commands", f"{prompt_name}.txt")
SYSTEM_PROMPT = load_prompt_template(prompt_file)

IS_COT_PROMPT = "cot" in prompt_name.lower() or prompt_name == "react_commands"

VALID_LABELS = [
    "BALANCE NOT UPDATED AFTER BANK TRANSFER",
    "BALANCE NOT UPDATED AFTER CHEQUE OR CASH DEPOSIT",
    "CARD PAYMENT FEE CHARGED",
    "CASH WITHDRAWAL CHARGE",
    "DECLINED CASH WITHDRAWAL",
    "DIRECT DEBIT PAYMENT NOT RECOGNISED",
    "TRANSACTION CHARGED TWICE",
    "TRANSFER FEE CHARGED",
    "TRANSFER NOT RECEIVED BY RECIPIENT",
    "WRONG AMOUNT OF CASH RECEIVED",
]


# -----------------------------
# NORMALIZATION
# -----------------------------
def normalize_text(text):
    text = str(text).upper().strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


# -----------------------------
# MODEL CALL
# -----------------------------
def query(command_text):
    """Send one banking customer query to the local model."""
    start_time = time.time()

    try:
        if IS_COT_PROMPT:
            user_prompt = (
                "Classify this banking customer-support query using the required reasoning format.\n\n"
                f"Customer query:\n{command_text}\n\n"
            )

            options = {
                "temperature": 0.0,
                "num_predict": 160,
            }

        else:
            user_prompt = (
                "Classify this banking customer-support query.\n"
                "Return only one bracketed intent label.\n\n"
                f"Customer query:\n{command_text}\n\n"
                "Label:"
            )

            options = {
                "temperature": 0.0,
                "num_predict": 35,
                "stop": ["\n"],
            }

        response = client.generate(
            model=MODEL_NAME,
            system=SYSTEM_PROMPT,
            prompt=user_prompt,
            options=options,
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
    text = normalize_text(raw_text)

    # Prefer bracketed output
    if is_bracketed:
        bracket_matches = re.findall(r"\[(.*?)\]", text)

        if bracket_matches:
            last_bracket = normalize_text(bracket_matches[-1])

            for label in VALID_LABELS:
                if last_bracket == label:
                    return label

            # fallback: bracket contains the label plus some extra text
            for label in VALID_LABELS:
                if label in last_bracket:
                    return label

    # Fallback: label appears anywhere in output
    for label in VALID_LABELS:
        if label in text:
            return label

    return "UNKNOWN"


# -----------------------------
# EVALUATION LOOP
# -----------------------------
def run_evaluation(dataset_path, output_csv_path, bracketed_output=True):
    print(f"\n🚀 Starting Commands evaluation on: {dataset_path}")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {prompt_name}")
    print(f"CoT mode: {IS_COT_PROMPT}")

    df = pd.read_excel(dataset_path)

    predictions = []
    latencies = []

    for idx, row in df.iterrows():
        command_text = row["review_text"]

        print("--------------------------------------------------------------------------------")
        print(f"customer query to classify : {command_text}")

        raw_pred, latency = query(command_text)
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
    df["predicted_intent"] = predictions
    df["latency_seconds"] = latencies

    os.makedirs("results", exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    # -----------------------------
    # METRICS
    # -----------------------------
    valid_mask = df["predicted_intent"].isin(VALID_LABELS)
    filtered_df = df[valid_mask]

    if len(filtered_df) == 0:
        print("❌ No valid predictions found. Check model output and clean_prediction().")
        return

    acc = accuracy_score(
        filtered_df["ground_truth"],
        filtered_df["predicted_intent"]
    )

    avg_latency = filtered_df["latency_seconds"].mean()
    unknown_count = len(df) - len(filtered_df)

    report_string = classification_report(
        filtered_df["ground_truth"],
        filtered_df["predicted_intent"],
        labels=VALID_LABELS,
        zero_division=0
    )

    strict_accuracy = accuracy_score(
        df["ground_truth"],
        df["predicted_intent"].where(df["predicted_intent"].isin(VALID_LABELS), "UNKNOWN")
    )

    report_text = (
        f"================== COMMANDS PERFORMANCE REPORT ==================\n"
        f"Dataset: {os.path.basename(dataset_path)}\n"
        f"Prompt: {prompt_name}\n"
        f"Model: {MODEL_NAME}\n"
        f"Valid Accuracy: {acc * 100:.2f}%\n"
        f"Strict Accuracy Including UNKNOWN: {strict_accuracy * 100:.2f}%\n"
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
    commands_path = os.path.join("datasets", "dataset_commands.xlsx")

    run_evaluation(
        dataset_path=commands_path,
        output_csv_path=os.path.join(
            "results",
            f"{prompt_name}_{MODEL_SHORT_NAME}_commands.csv"
        ),
        bracketed_output=True
    )