from datasets import Audio, load_dataset, Dataset, load_from_disk, disable_caching
from pydub import AudioSegment
from pathlib import Path
import os
import openai
from tqdm import tqdm, trange
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import DataLoader
from numpy import array
import csv
import json
import time

# enter your openai api key
TOKEN = YOUR_OPENAI_API_TOKEN

# select the task for testing dataset
all_datasets = [
#    'DynamicSuperb/AccentClassification_AccentdbExtended', 
#    'DynamicSuperb/BirdSoundDetection_Warblrb10k', 
#    'DynamicSuperb/ChordClassification_AcousticGuitarAndPiano', 
#    'DynamicSuperb/DialogueActClassification_DailyTalk', 
#    'DynamicSuperb/DialogueActPairing_DailyTalk',
#    'DynamicSuperb/DialogueEmotionClassification_DailyTalk', 
#    'DynamicSuperb/EmotionRecognition_MultimodalEmotionlinesDataset', 
#    'DynamicSuperb/EnhancementDetection_LibrittsTestCleanWham', 
#    'DynamicSuperb/EnvironmentalSoundClassification_AnimalsESC50', 
#    'DynamicSuperb/EnvironmentalSoundClassification_ExteriorAndUrbanNoisesESC50', 
#    'DynamicSuperb/EnvironmentalSoundClassification_HumanAndNonSpeechSoundsESC50', 
#    'DynamicSuperb/EnvironmentalSoundClassification_InteriorAndDomesticSoundsESC50', 
#    'DynamicSuperb/EnvironmentalSoundClassification_NaturalSoundscapesAndWaterSoundsESC50',  
#    'DynamicSuperb/Intent_Classification_FluentSpeechCommands_Action',
#    'DynamicSuperb/Intent_Classification_FluentSpeechCommands_Location',
#    'DynamicSuperb/Intent_Classification_FluentSpeechCommands_Object', 
#    'DynamicSuperb/NoiseDetectiongaussian_LJSpeechMusan', 
#    'DynamicSuperb/NoiseDetectionmusic_LJSpeechMusan', 
#    'DynamicSuperb/NoiseDetectionnoise_LJSpeechMusan', 
#    'DynamicSuperb/NoiseDetectionspeech_LJSpeechMusan', 
#    'DynamicSuperb/ReverberationDetectionlargeroom_LJSpeechRirsNoises', 
#    'DynamicSuperb/ReverberationDetectionlargeroom_VCTKRirsNoises', 
#    'DynamicSuperb/ReverberationDetectionmediumroom_LJSpeechRirsNoises', 
#    'DynamicSuperb/ReverberationDetectionmediumroom_VCTKRirsNoises', 
#    'DynamicSuperb/ReverberationDetectionsmallroom_LJSpeechRirsNoises', 
#    'DynamicSuperb/ReverberationDetectionsmallroom_VCTKRirsNoises', 
#    'DynamicSuperb/SarcasmDetection_Mustard', 
#    'DynamicSuperb/SpeakerCounting_LibriTTSTestClean', 
#    'DynamicSuperb/SpeakerVerification_LibriSpeechTestClean',
#    'DynamicSuperb/SpeakerVerification_VCTK',
#    'DynamicSuperb/SpeechCommandRecognition_GoogleSpeechCommandsV1', 
#    'DynamicSuperb/SpeechDetection_LJSpeech', 
#    'DynamicSuperb/SpeechDetection_LibriSpeechTestClean', 
#    'DynamicSuperb/SpeechDetection_LibriSpeechTestOther', 
#    'DynamicSuperb/SpokenTermDetection_LJSpeech', 
#    'DynamicSuperb/SpokenTermDetection_LibriSpeechTestClean', 
#    'DynamicSuperb/SpokenTermDetection_LibriSpeechTestOther', 
#    'DynamicSuperb/SpoofDetection_ASVspoof2015',
#    'DynamicSuperb/SpoofDetection_ASVspoof2017', 
#    'DynamicSuperb/StressDetection_MIRSD', 
#    'DynamicSuperb/SpeechTextMatching_LJSpeech', 
#    'DynamicSuperb/SpeechTextMatching_LibriSpeechTestClean', 
#    'DynamicSuperb/SpeechTextMatching_LibriSpeechTestOther',
#    'DynamicSuperb/MultiSpeakerDetection_LibriSpeechTestClean',
#    'DynamicSuperb/MultiSpeakerDetection_VCTK'
#    'DynamicSuperb/NoiseDetectiongaussian_VCTKMusan',
#    'DynamicSuperb/NoiseDetectionmusic_VCTKMusan',
#    'DynamicSuperb/NoiseDetectionnoise_VCTKMusan', 
#    'DynamicSuperb/NoiseDetectionspeech_VCTKMusan', 
#    'DynamicSuperb/NoiseSNRLevelPredictiongaussian_VCTKMusan', 
#    'DynamicSuperb/NoiseSNRLevelPredictionmusic_VCTKMusan', 
#    'DynamicSuperb/NoiseSNRLevelPredictionnoise_VCTKMusan', 
#    'DynamicSuperb/NoiseSNRLevelPredictionspeech_VCTKMusan',
#    'DynamicSuperb/HowFarAreYou_3DSpeaker', 
#    'DynamicSuperb/LanguageIdentification_VoxForge',
]

disable_caching()

def query_whisper(audio, processor, model):
    input_features = processor(audio["array"], sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features.to("cuda"))
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription

def query_chatgpt(text, prompt, model_name="gpt-3.5-turbo"):
    openai.api_key = TOKEN

    while True:
        try:
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ]
            )
            break
        except:
            print("\nChatGPT API failed. Try again.\n")
            time.sleep(5)
            continue


    response = completion.choices[0].message["content"].encode('unicode-escape').decode('unicode-escape')
    return response

def main():
    # path for caching the downloaded dataset
    data_path = Path("./dynamic-superb-dataset-test")
    # path for write down the asr transcription
    transcript_path = Path("./whisper_transcript")
    # path for write down the chatgpt response
    response_path = Path("./chatgpt_response")
    # path for logging the final result
    output_result_log = Path("./result")

    # Whisper large v2
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to("cuda")
    model.config.forced_decoder_ids = None

    for dataset_name in tqdm(all_datasets):
        print(f"Load {dataset_name}")

        # Download dataset to disk
        if (data_path/dataset_name).exists():
            print("Dataset Exists")
            dataset = load_from_disk(data_path/dataset_name)
            print("Dataset loaded")
        else:
            dataset = load_dataset(dataset_name, cache_dir=data_path)
            (data_path/dataset_name).mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(data_path/dataset_name)    
        
        
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        # if the task has more than 1 audio
        #dataset = dataset.cast_column("audio2", Audio(sampling_rate=16000))
        print(dataset)
        print(dataset["test"][0]["instruction"])
        
        ######### Can be take out the comment for Inferencing Whisper ########
        with open(f"{transcript_path}/{dataset_name.split('/')[1]}_transcript_3.txt", 'w') as f:
            # if the task has more than 1 audio
            #for audio_data, audio_data_2 in zip(tqdm(dataset["test"]["audio"]), tqdm(dataset["test"]["audio2"])):
            
            # if the task has only 1 audio
            for i, audio_data in tqdm(enumerate(dataset["test"]["audio"]), total=len(dataset["test"])):
                transcript = query_whisper(audio_data, processor, model)
                
                # if the task has more than 1 audio
                #transcript_2 = query_whisper(audio_data_2, processor, model)

                # if the task has only 1 audio
                f.writelines(t + '\n' for t in transcript)

                # if the task has more than 1 audio
                #f.writelines(t + "|" + t2 + '\n' for t, t2 in zip(transcript, transcript_2))
        ######################################################################

        ######### Can be take out the comment for Querying ChatGPT ###########
        model_name="gpt-3.5-turbo"
        answers = []
        err = 0
        total = 0
        with open(f"{response_path}/{dataset_name.split('/')[1]}_chatgpt_response.txt", 'r+') as ans_file:
            with open(f"{transcript_path}/{dataset_name.split('/')[1]}_transcript.txt", 'r') as trans_file:
                answered = ans_file.readlines()
                lines = trans_file.readlines()
                for i, transcript in tqdm(enumerate(lines), total=len(lines)):
                    # if the task has only 1 instruction as text input
                    prompt = dataset["test"]["instruction"][i] + " Choose one answer from above options. Only one word is needed."
                    
                    # if the task has additional text input
                    #prompt = f"The given word is {dataset['test']['text'][i]} " + dataset["test"]["instruction"][i] + " Choose one answer from above options. Only one word is needed."
                    
                    label = dataset["test"]["label"][i]

                    if i < len(answered):
                        total += 1
                        if answered[i].lower() != label.lower():
                            err += 1
                        continue

                    # if the task has more than 1 audio
                    #s1, s2 = transcript.split("|")[0], transcript.split("|")[1]
                    #transcript = f"The first utterance is \"{s1}\", and the second one is \"{s2}\"."

                    ans = query_chatgpt(transcript, prompt, model_name)
                    answers.append(ans)
                    ans_file.writelines(ans + '\n')

                    total += 1
                    if ans.lower() != label.lower():
                        err += 1
        
        print("Testing Finish!")
        print("Dataset Name: " + dataset_name)
        error_rate = err / total
        accuracy = 1.0 - error_rate
        print("Error Rate(error samples/ total samples): " + str(err/total) + f"({err}/{total})")

        results = {}
        results[dataset_name] = {
            "Accuracy": accuracy,
            "Error-rate": error_rate,
            "Correct num": (total - err),
            "Error num": err,
            "Total num": total
        }

        with open(f"{output_result_log}/{dataset_name.split('/')[1]}.json", "w") as fp:
            json.dump(results, fp, indent=4)
        ######################################################################

if __name__ == "__main__":
    main()
        



