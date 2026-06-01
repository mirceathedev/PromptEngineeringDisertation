# Prompt Engineering on local LLMs across multiple datasets.

This repository contains experiments for evaluating prompt engineering techniques on local LLMs using Ollama.

## Project Story

The initial goal of this project was to evaluate how much prompt engineering can improve the performance of local Large Language Models on text classification tasks.

The first dataset tested was **SST-5**, a fine-grained sentiment classification dataset with five sentiment classes:

* Very Negative
* Negative
* Neutral
* Positive
* Very Positive

Several prompting strategies were tested on SST-5, including zero-shot prompting, few-shot prompting, chain-of-thought prompting, self-consistency, polarity/intensity decomposition, and ReAct-style prompting.

However, the best accuracy obtained on SST-5 was around **55%**, even after multiple prompt refinements. The main difficulty was that SST-5 is highly subjective and fine-grained. Most errors came from confusing similar sentiment intensities, especially:

* Positive vs Very Positive
* Negative vs Very Negative
* Neutral vs weak positive/negative reviews

Because the improvement margin on SST-5 was limited, the project shifted toward a more structured intent-classification task where prompt engineering could be evaluated more clearly.

The second dataset used was a banking customer-support intent dataset. From the full dataset, only the **10 intent groups with the highest number of available samples** were selected. This created a balanced 10-class classification benchmark with 100 examples per class.

The selected banking intents were:

* Balance not updated after bank transfer
* Balance not updated after cheque or cash deposit
* Card payment fee charged
* Cash withdrawal charge
* Declined cash withdrawal
* Direct debit payment not recognised
* Transaction charged twice
* Transfer fee charged
* Transfer not received by recipient
* Wrong amount of cash received

This dataset was better suited for testing prompt engineering because the labels are semantically close but still clearly distinguishable with good instructions. For example, the model needs to separate:

* balance not updated after a transfer vs recipient did not receive the transfer
* cash withdrawal charge vs declined cash withdrawal vs wrong cash amount received
* card payment fee vs transfer fee vs cash withdrawal fee

The final experiments compared local 3B models such as **Llama 3.2 3B** and **Qwen 2.5 3B Instruct** across several prompting strategies:

* zero-shot base prompting
* guided zero-shot prompting
* few-shot prompting
* zero-shot chain-of-thought prompting
* few-shot chain-of-thought prompting
* ReAct-style prompting

The best result so far was obtained on the banking intent dataset using **few-shot chain-of-thought prompting**, where Qwen 2.5 3B Instruct reached around **90% strict accuracy**.

This shows that prompt engineering can significantly improve local LLM classification performance, especially when the task has clear intent labels and the prompt explicitly teaches the model how to distinguish similar classes.




## Models tested

- llama3.2:3b
- qwen2.5:3b-instruct

## Datasets

- SST-5 sentiment classification
- Banking/Commands intent classification


## Prompting strategies

- Zero-shot
- Guided zero-shot
- Few-shot
- Zero-shot Chain-of-Thought
- Few-shot Chain-of-Thought
- ReAct-style prompting