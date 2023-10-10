# ASR-ChatGPT

The repository shows the implementation of ASR-ChatGPT baseline in [our work](https://arxiv.org/abs/2309.09510) [[1]](https://arxiv.org/abs/2309.09510). 

## How to Run

### Enviroment
Make shure the packages mentioned in the script have been installed in your machine. 

### OpenAI API Key
Enter your OpenAI API Key to **line 15** in `asr_chatgpt.py`

```
TOKEN = YOUR_OPENAI_API_TOKEN
```

### Select the Tasks for Testing
The list in **line 18** in `asr_chatgpt.py` shows the tasks for testing in Dynamic-SUPERB(original version). You can select them and download for inferencing. 

```
all_datasets = [
#    'DynamicSuperb/AccentClassification_AccentdbExtended', 
#    'DynamicSuperb/BirdSoundDetection_Warblrb10k', 
    ...
]
```

### Check the Path of Directories
You can modify the related directory path by entering your new path to **line 108 to 114** in `asr_chatgpt.py`

```
# path for caching the downloaded dataset
data_path = Path("./dynamic-superb-dataset-test")
# path for write down the asr transcription
transcript_path = Path("./whisper_transcript")
# path for write down the chatgpt response
response_path = Path("./chatgpt_response")
# path for logging the final result
output_result_log = Path("./result")
```

### Check Out the Settings 
There are some special settings for some specific tasks. Watch out the comment in `asr_chatgpt.py`. You can comment or release them. The default setting is for only 1 audio and 1 instruction for input. 
Special settings such as **line 136**
```
# if the task has more than 1 audio
dataset = dataset.cast_column("audio2", Audio(sampling_rate=16000))
```
or **line 173**
```
# if the task has additional text input
prompt = f"The given word is {dataset['test']['text'][i]} " + dataset["test"]["instruction"][i] + " Choose one answer from above options. Only one word is needed."
```

### Run the Script
Type the below command to your terminal and run.
```
python3 asr_chatgpt.py
```

## Reference
[[1]](https://arxiv.org/abs/2309.09510) Huang, Chien-yu, et al. "Dynamic-SUPERB: Towards A Dynamic, Collaborative, and Comprehensive Instruction-Tuning Benchmark for Speech." arXiv preprint arXiv:2309.09510 (2023).