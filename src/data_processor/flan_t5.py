# -*- encoding: utf-8 -*-

class FlanT5Train:
    def __init__(self, ds, tokenizer, max_length=2048):
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        result = self.tokenizer(examples['instruction'], max_length=self.max_length)
        return {
            'input_ids': result['input_ids'],
            'attention_mask': result['attention_mask'],
            'labels': examples['output']
        }