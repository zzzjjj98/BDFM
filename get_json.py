import json
import os
import random


def create_json_for_brats(data_dir, output_filename):
    filedict = {"training": [], "validation": []}
    training_contents = []
    val_contents = []

    # Training data
    train_dir = os.path.join(data_dir, "TrainingData")
    if os.path.exists(train_dir):
        for filename in os.listdir(train_dir):
            modalities = ["t2f", "t1c", "t1n", "t2w"]
            for mod in modalities:
                training_contents.append({
                    "image": f"TrainingData/{filename}/{filename}-{mod}.nii.gz",
                    "label": f"TrainingData/{filename}/{filename}-seg.nii.gz"
                })
        filedict["training"] = training_contents

    # Validation data
    val_dir = os.path.join(data_dir, "ValidationData")
    if os.path.exists(val_dir):
        for filename in os.listdir(val_dir):
            modalities = ["t2f", "t1c", "t1n", "t2w"]
            for mod in modalities:
                val_contents.append({
                    "image": f"ValidationData/{filename}/{filename}-{mod}.nii.gz"
                })
        filedict["validation"] = val_contents

    # Write to JSON file
    with open(output_filename, "w") as f:
        json.dump(filedict, f, indent=4)


def create_json_for_dataset(data_dir, output_filename):
    filedict = {"training": [], "validation": []}
    training_contents = []
    val_contents = []

    # Training data
    train_dir = os.path.join(data_dir, "TrainingData")
    if os.path.exists(train_dir):
        for filename in os.listdir(train_dir):
            training_contents.append({"image": f"TrainingData/{filename}"})
        filedict["training"] = training_contents

    # Validation data
    val_dir = os.path.join(data_dir, "ValidationData")
    if os.path.exists(val_dir):
        for filename in os.listdir(val_dir):
            val_contents.append({"image": f"ValidationData/{filename}"})
        filedict["validation"] = val_contents

    # Write to JSON file
    with open(output_filename, "w") as f:
        json.dump(filedict, f, indent=4)


datasets_brats = [
        ("/mnt/data/datasets/BraTS23_MEN", "BraTS23_MEN_onemodal.json"),
        ("/mnt/data/datasets/BraTS23_MET", "BraTS23_MET_onemodal.json"),
        ("/mnt/data/datasets/BraTS23_PED", "BraTS23_PED_onemodal.json"),
        ("/mnt/data/datasets/BraTS23_GLI", "BraTS23_GLI_onemodal.json")
]

datasets_others = [
        ("/mnt/data/datasets/AtlasR2", "AtlasR2.json"),
        ("/mnt/data/datasets/ISLES2022", "ISLES2022.json"),
        ("/mnt/data/datasets/BrainPTM2021", "BrainPTM2021.json"),
        ("/mnt/data/datasets/OASIS", "OASIS.json")
]


# Process datasets
for data_dir, output_filename in datasets_brats:
    create_json_for_brats(data_dir, output_filename)

for data_dir, output_filename in datasets_others:
    create_json_for_dataset(data_dir, output_filename)

