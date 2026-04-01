#!/usr/bin/env python3
"""Fine-tune Nemotron-3-Nano-30B on self-modification transcripts.

Uses BF16 model + QLoRA (4-bit NF4 via bitsandbytes) + PEFT LoRA + TRL SFTTrainer.
- BF16 model quantized to NF4 at load time (~60GB → ~15GB in memory)
- LoRA rank 8, targets attention + Mamba + MoE projections
- Response-only training via DataCollatorForCompletionOnlyLM

Usage:
  python3 train.py --data training_data.jsonl
"""

import argparse
import json
import os
import sys


def load_training_data(data_path: str) -> list[dict]:
    """Load training data from JSONL file."""
    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} training examples from {data_path}")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Nemotron on self-modification data")
    parser.add_argument("--data", default="training_data.jsonl",
                        help="Path to training data JSONL")
    parser.add_argument("--model-name",
                        default="/workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                        help="Base model path or HF name")
    parser.add_argument("--output-dir", default="./lora_output",
                        help="Directory to save LoRA adapter")
    parser.add_argument("--max-seq-length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Maximum training steps (0 = use num_epochs)")
    parser.add_argument("--num-epochs", type=int, default=3,
                        help="Number of epochs (used if max_steps=0)")
    parser.add_argument("--warmup-steps", type=int, default=10,
                        help="Warmup steps")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load model and data but don't train")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

    # Set TRITON path for DGX Spark
    if "TRITON_PTXAS_PATH" not in os.environ:
        ptxas = "/usr/local/cuda/bin/ptxas"
        if os.path.exists(ptxas):
            os.environ["TRITON_PTXAS_PATH"] = ptxas

    # ---------------------------------------------------------------
    # 1. Load model
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Loading model: {args.model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load BF16 model with 4-bit NF4 quantization via bitsandbytes.
    # This quantizes on-the-fly during loading (~60GB bf16 → ~15GB NF4 in memory).
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={"": 0},
        attn_implementation="eager",
    )

    # Verify NF4 quantization actually activated
    mem_after_load = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU memory after model load: {mem_after_load:.1f} GB")
    if mem_after_load > 30:
        print(f"WARNING: Memory {mem_after_load:.1f}GB suggests NF4 quantization "
              f"may not have activated (expected ~15GB)")
        if mem_after_load > 80:
            print("ERROR: Model loaded as full bf16. Aborting to prevent OOM.")
            sys.exit(1)

    # Prepare for QLoRA training — freezes base, casts non-quantized params to fp32
    # Skips Params4bit (NF4 quantized weights stay quantized)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    print(f"Model loaded: {type(model).__name__}")
    print(f"Device map: {getattr(model, 'hf_device_map', 'N/A')}")
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU memory after model load: {mem_gb:.1f} GB")

    # ---------------------------------------------------------------
    # 2. Apply LoRA
    # ---------------------------------------------------------------
    print(f"\nApplying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")

    # Find which target modules actually exist in this model
    candidate_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",     # Attention
        "qkv_proj",                                   # Fused attention (Nemotron)
        "gate_proj", "up_proj", "down_proj",           # MoE FFN
        "in_proj", "out_proj",                         # Mamba
    ]
    model_modules = {name.split(".")[-1] for name, _ in model.named_modules()}
    target_modules = [m for m in candidate_modules if m in model_modules]
    print(f"  Target modules: {target_modules}")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------------------------------------------------------------
    # 3. Load and format training data
    # ---------------------------------------------------------------
    print(f"\nLoading training data from {args.data}")
    raw_examples = load_training_data(args.data)

    formatted = []
    skipped = 0
    for example in raw_examples:
        convos = example["conversations"]
        try:
            text = tokenizer.apply_chat_template(
                convos,
                tokenize=False,
                add_generation_prompt=False,
            )
            tokens = tokenizer.encode(text)
            if len(tokens) <= args.max_seq_length:
                formatted.append({"text": text})
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Skipped example: {e}")

    print(f"Formatted: {len(formatted)}, Skipped: {skipped}")

    if not formatted:
        print("ERROR: No valid training examples.")
        sys.exit(1)

    dataset = Dataset.from_list(formatted)
    print(f"Dataset: {len(dataset)} examples")

    print(f"\n--- Sample (first 500 chars) ---")
    print(dataset[0]["text"][:500])
    print("---")

    if args.dry_run:
        print("\nDry run complete. Model and data loaded successfully.")
        print(f"Would train for {args.max_steps} steps with batch={args.batch_size}")
        return

    # ---------------------------------------------------------------
    # 4. Configure trainer with response-only masking
    # ---------------------------------------------------------------
    print(f"\nConfiguring SFTTrainer")

    # Response-only: only compute loss on assistant responses
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    training_args = SFTConfig(
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.num_epochs if args.max_steps <= 0 else 1,
        learning_rate=args.lr,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=args.output_dir,
        report_to="none",
        bf16=True,
        save_steps=50,
        save_total_limit=3,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=training_args,
        data_collator=collator,
    )

    # Live loss logging callback
    class LogLossCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                mem = torch.cuda.memory_allocated() / 1024**3
                print(f"step {state.global_step:4d} | loss {logs['loss']:.4f} | "
                      f"lr {logs.get('learning_rate', 0):.6f} | "
                      f"grad_norm {logs.get('grad_norm', 0):.3f} | "
                      f"mem {mem:.1f}GB", flush=True)

    trainer.add_callback(LogLossCallback())

    # ---------------------------------------------------------------
    # 5. Train
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Starting training")
    print(f"  Batch size: {args.batch_size} x {args.grad_accum} = "
          f"{args.batch_size * args.grad_accum} effective")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Sequence length: {args.max_seq_length}")
    print(f"{'='*60}\n")

    gpu_stats = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu_stats.name}, {gpu_stats.total_memory / 1024**3:.1f} GB")

    trainer_stats = trainer.train()

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Total steps: {trainer_stats.global_step}")
    print(f"  Final loss: {trainer_stats.training_loss:.4f}")
    print(f"{'='*60}")

    # ---------------------------------------------------------------
    # 6. Save LoRA adapter
    # ---------------------------------------------------------------
    print(f"\nSaving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "base_model": args.model_name,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "max_steps": args.max_steps,
        "training_examples": len(dataset),
        "final_loss": trainer_stats.training_loss,
        "total_steps": trainer_stats.global_step,
    }
    with open(os.path.join(args.output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! To use with vLLM:")
    print(f"  vllm serve {args.model_name} --enable-lora \\")
    print(f"    --lora-modules neuroplastic={args.output_dir}")


if __name__ == "__main__":
    main()
