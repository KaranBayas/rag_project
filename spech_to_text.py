import whisper
import json 
import os

model = whisper.load_model("base") # loading base model, you can also use "small", "medium", "large" depending on your needs
audios = os.listdir("videos");




for audio in audios:
    title = audio.split(" [")[0]

    result = model.transcribe("videos/"+audio,
                           language="hi",
                           task="translate",
                           word_timestamps=False )
    chunks =[]
    for segment in result["segments"]:
        chunks.append({"title":title, "start": segment["start"], 
                       "end": segment["end"],"text": segment["text"]})
    
    chunks_with_metadata = {"chunks":chunks,"text":result["text"]}

    with open(f"jsons/{title}.json", "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)

    print(f"{title} videos processed successfully!")