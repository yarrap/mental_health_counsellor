import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from finetune import FineTuner  

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLaMA instruct model on a math dataset using a single data file"
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model name or path for the LLaMA instruct model")##required=True,
    parser.add_argument("--data_file", type=str, default = "/project/pi_hongyu_umass_edu/n_yarrabolu/personal/data/processed/train_processed.csv",  help="Path to the JSON file containing the math dataset")##required=True,
    parser.add_argument("--output_dir", type=str,  default = "/project/pi_hongyu_umass_edu/n_yarrabolu/personal/results/", help="Directory where the fine-tuned model will be saved")##required=True,
    parser.add_argument("--text_column", type=str, default="questionText", help="Name of the text column in the JSON file")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum token length for tokenization")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size per device")
    parser.add_argument("--logging_steps", type=int, default=100, help="Number of steps between logging")
    parser.add_argument("--eval_steps", type=int, default=500, help="Number of steps between evaluation")
    parser.add_argument("--save_steps", type=int, default=500, help="Number of steps between saving checkpoints")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj", help="Comma-separated target modules for LoRA")
    
    args = parser.parse_args()
    # import pdb;pdb.set_trace()

    finetuner = FineTuner(args)
    finetuner.load_data()       
    finetuner.tokenize_data()   
    finetuner.train()           

if __name__ == "__main__":
    main()