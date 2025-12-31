from datasets import load_dataset


ds = load_dataset("Miguel231/refCOCOg_in_tar", split="train")
# ['jpg', 'txt', 'json', '__key__', '__url__']

print(ds[0]['jpg'])
