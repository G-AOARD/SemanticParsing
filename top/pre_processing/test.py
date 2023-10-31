import json

data = json.load(open("./outputs/top/trainer_with_SemRoberta_with_grammars_seed_0/dev_logs/dev_log_epoch_9.json", encoding="utf-8"))

total = len(data)
correct = 0
for item in data:
    if item["pred"] != item["label"]:
        print("---------------------")
        print(item)
        correct += 1

print(f"Acc: {correct/total}")