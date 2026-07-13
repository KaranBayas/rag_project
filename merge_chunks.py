import os
import json

json_folder = "jsons"
group_size = 5

for filename in os.listdir(json_folder):

    if not filename.endswith(".json"):
        continue

    filepath = os.path.join(json_folder, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    old_chunks = data["chunks"]
    new_chunks = []

    for i in range(0, len(old_chunks), group_size):

        group = old_chunks[i:i + group_size]

        merged_chunk = {
            "title": group[0]["title"],
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": " ".join(chunk["text"].strip() for chunk in group)
        }

        new_chunks.append(merged_chunk)

    data["chunks"] = new_chunks

    # Optional: regenerate the full transcript
    data["text"] = " ".join(chunk["text"] for chunk in new_chunks)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ {filename}: {len(old_chunks)} → {len(new_chunks)} chunks")