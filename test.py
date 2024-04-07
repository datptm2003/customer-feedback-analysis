import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

num_labels = 3
id2label = {
    0: "NEG",
    1: "NEU",
    2: "POS"
}
label2id = {
    "NEG": 0,
    "NEU": 1,
    "POS": 2
}
config = AutoConfig.from_pretrained("vinai/phobert-base-v2", num_labels=num_labels, id2label=id2label, label2id=label2id)
base_model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base-v2",config=config)
model = PeftModel.from_pretrained(base_model, "datptm2003/lora-vietnamese-feedback-analysis")

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

texts = [
    "Hàng rất tốt nha, cảm ơn shop.",
    "Áo tạm được, chất lượng vừa phải",
    "Hàng kém chất lượng, không nên mua",
    "Wow tuyệt quá!",
    "Áo rất mát, mua rất đáng tiền",
    "Phí tiền, khuyên các bạn nên tẩy chay shop.",
    "Quá xấu",
    "Màu k giống trong hình,Chất mặc nóng, bí không thấm hút mồ hôi.",
    "Chất da cá ( tiền naoc của đó ).",
    "Quần chưa thử nên không biết có che được không nhưng mình k thích cái viền ren lắm.",
    "Bình thường, không quá tệ"
]

for text in texts:
    inputs = tokenizer.encode(text, return_tensors="pt")

    logits = model(inputs).logits
    # print("---Logits:",logits)
    predictions = torch.max(logits,1).indices

    id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
    print(f"\"{text}\": \n>>> {id2label[predictions.tolist()[0]]}\n")