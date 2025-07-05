from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import torch
import evaluate
import numpy as np
from huggingface_hub import login

# 🔐 Step 0: Login to Hugging Face Hub
login(token="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")  # replace with your actual token

# ✅ Step 1: Load the Food-101 dataset
dataset = load_dataset("ethz/food101")
labels = dataset["train"].features["label"].names

# ✅ Step 2: Load processor and define transforms
checkpoint = "google/vit-base-patch16-224-in21k"
processor = AutoImageProcessor.from_pretrained(checkpoint)

normalize = Normalize(mean=processor.image_mean, std=processor.image_std)

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    normalize,
])

# ✅ Step 3: Custom transform wrapper
def transform_examples(example_batch):
    example_batch["pixel_values"] = [transform(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

# Apply transform
prepared_ds = dataset.cast_column("image", dataset["train"].features["image"])
prepared_ds = prepared_ds.with_transform(transform_examples)

# ✅ Step 4: Load model
model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label={str(i): label for i, label in enumerate(labels)},
    label2id={label: str(i) for i, label in enumerate(labels)},
)

# ✅ Step 5: Metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

# ✅ Step 6: Training Arguments
training_args = TrainingArguments(
    output_dir="./vit-food101-model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    logging_steps=50,
    remove_unused_columns=False,
    push_to_hub=True,
    hub_model_id="pavankumar550/food101-vit-custom",
    hub_strategy="every_save",
    report_to="none",
)

# ✅ Step 7: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],  # ✅ FIXED: 'test' ➝ 'validation'
    tokenizer=processor,
    compute_metrics=compute_metrics,
)

# ✅ Step 8: Train and Push to Hub
trainer.train()

# Save and Push (explicit)
trainer.save_model("./vit-food101-model")
trainer.push_to_hub()


##WOrking -02

# from datasets import load_dataset
# from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from PIL import Image
# import torch
# import evaluate
# import numpy as np
# from huggingface_hub import login

# # 🔐 Step 0: Login to Hugging Face Hub
# login(token="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")  # replace with your token

# # ✅ Step 1: Load the Food-101 dataset
# dataset = load_dataset("food101")
# labels = dataset["train"].features["label"].names

# # ✅ Step 2: Load processor and define transforms
# checkpoint = "google/vit-base-patch16-224-in21k"
# processor = AutoImageProcessor.from_pretrained(checkpoint)

# normalize = Normalize(mean=processor.image_mean, std=processor.image_std)

# transform = Compose([
#     Resize((224, 224)),
#     ToTensor(),
#     normalize,
# ])

# # ✅ Step 3: Custom transform wrapper
# def transform_examples(example_batch):
#     example_batch["pixel_values"] = [transform(image.convert("RGB")) for image in example_batch["image"]]
#     return example_batch

# # Apply transform
# prepared_ds = dataset.cast_column("image", dataset["train"].features["image"])
# prepared_ds = prepared_ds.with_transform(transform_examples)

# # ✅ Step 4: Load model
# model = AutoModelForImageClassification.from_pretrained(
#     checkpoint,
#     num_labels=len(labels),
#     id2label={str(i): label for i, label in enumerate(labels)},
#     label2id={label: str(i) for i, label in enumerate(labels)},
# )

# # ✅ Step 5: Metric
# accuracy = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = np.argmax(logits, axis=1)
#     return accuracy.compute(predictions=preds, references=labels)

# # ✅ Step 6: Training Arguments
# training_args = TrainingArguments(
#     output_dir="./vit-food101-model",
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     num_train_epochs=5,
#     logging_steps=50,
#     remove_unused_columns=False,
#     push_to_hub=True,
#     hub_model_id="pavankumar550/food101-vit-custom",  # change to your Hugging Face username/model
#     hub_strategy="every_save",
#     report_to="none",
# )

# # ✅ Step 7: Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=prepared_ds["train"],
#     eval_dataset=prepared_ds["test"],
#     tokenizer=processor,
#     compute_metrics=compute_metrics,
# )

# # ✅ Step 8: Train and Push to Hub
# trainer.train()

# # Save and Push (explicit)
# trainer.save_model("./vit-food101-model")
# trainer.push_to_hub()


# from datasets import load_dataset
# from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# from torchvision.transforms.functional import to_pil_image
# from PIL import Image
# import torch
# import evaluate
# import numpy as np
# import os
# from huggingface_hub import login

# # 🔐 Step 0: Login to Hugging Face Hub (Paste your token here)
# login(token="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")

# # ✅ Step 1: Load the Food-101 dataset
# dataset = load_dataset("ethz/food101")
# labels = dataset["train"].features["label"].names

# # ✅ Step 2: Load processor and define transforms
# checkpoint = "google/vit-base-patch16-224-in21k"
# processor = AutoImageProcessor.from_pretrained(checkpoint)

# normalize = Normalize(mean=processor.image_mean, std=processor.image_std)

# transform = Compose([
#     Resize((224, 224)),
#     ToTensor(),
#     normalize,
# ])

# # ✅ Step 3: Custom transform wrapper
# def transform_examples(example_batch):
#     example_batch["pixel_values"] = [transform(image.convert("RGB")) for image in example_batch["image"]]
#     return example_batch

# # Apply transform
# prepared_ds = dataset.cast_column("image", dataset["train"].features["image"])
# prepared_ds = prepared_ds.with_transform(transform_examples)

# # ✅ Step 4: Load model
# model = AutoModelForImageClassification.from_pretrained(
#     checkpoint,
#     num_labels=len(labels),
#     id2label={str(i): label for i, label in enumerate(labels)},
#     label2id={label: str(i) for i, label in enumerate(labels)},
# )

# # ✅ Step 5: Metric
# accuracy = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = np.argmax(logits, axis=1)
#     return accuracy.compute(predictions=preds, references=labels)

# # ✅ Step 6: Training Arguments

# training_args = TrainingArguments(
#     output_dir="./vit-food101-model",
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     num_train_epochs=5,
#     logging_steps=50,
#     remove_unused_columns=False,
#     push_to_hub=True,
#     hub_model_id="Elizah550/food101-vit-custom",
#     hub_strategy="every_save",
#     report_to="none",
# )

# # ✅ Step 7: Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=prepared_ds["train"],
#     eval_dataset=prepared_ds["test"],
#     tokenizer=processor,
#     compute_metrics=compute_metrics,
# )

# # ✅ Step 8: Train and Push
# trainer.train()

# # Save and Push (explicit)
# trainer.save_model("./vit-food101-model")
# trainer.push_to_hub()
