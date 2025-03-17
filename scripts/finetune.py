import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import torch



class FineTuner:
    def __init__(self, args) -> None:
        self.args = args

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.makedirs(args.output_dir, exist_ok=True)

        ##checking if cuda is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("No GPU available, using CPU")

        ## initialising tokeniser and setting pad token same as eod token if the pad token is not set up
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(args.model).to(self.device)

        ## defining lora 
        if args.use_lora:
            print("Using LoRA for fine-tuning")
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules.split(',')
            )
            self.model = get_peft_model(self.model, lora_config)

    '''loading dataset and creating prompt column with chat apply template format'''
    def load_data(self):
        dataset = load_dataset("csv", data_files=self.args.data_file, split="train")

        def apply_chat_template(dataset):
            messages = [
                {"role": "user", "content": dataset['Context']},
                {"role": "assistant", "content": dataset['Response']}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return {"prompt": prompt}
        dataset = dataset.map(apply_chat_template)
        self.dataset = dataset.train_test_split(test_size=0.1)
        print(f"Loaded dataset with {len(self.dataset['train'])} training examples and {len(self.dataset['test'])} test examples.")
        
    ''' Function for tokenisation of the prompt column from dataset'''
    def tokenize_data(self):
        def tokenize_function(example):
            tokens = self.tokenizer(
                example['prompt'], 
                padding="max_length", 
                truncation=True, 
                max_length=self.args.max_length,
                add_special_tokens=False
            )
            tokens['labels'] = [
                -100 if token == self.tokenizer.pad_token_id else token for token in tokens['input_ids']
            ]
            return tokens
        
        
        orig_columns = self.dataset["train"].column_names
        self.dataset["train"] = self.dataset["train"].map(tokenize_function, batched=True, remove_columns=orig_columns)
        self.dataset["test"] = self.dataset["test"].map(tokenize_function, batched=True, remove_columns=orig_columns)
        print(self.dataset["train"].shape)

    '''defining and training the model with the prepared dataset'''
    def train(self):
        print("Starting Training")
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.train_batch_size,
            per_device_eval_batch_size=self.args.eval_batch_size,
            logging_dir='./logs',
            logging_steps=self.args.logging_steps,
            eval_strategy="steps",
            eval_steps=self.args.eval_steps,
            save_steps=self.args.save_steps,
            learning_rate=self.args.learning_rate,
            save_total_limit=1,
            report_to="tensorboard"
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            data_collator=data_collator,
        )

        print("Training Model...")
        trainer.train()
        eval_results = trainer.evaluate()
        print("Finished Training")
        print("Evaluation Results:", eval_results)

        print("Saving Model...")
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)