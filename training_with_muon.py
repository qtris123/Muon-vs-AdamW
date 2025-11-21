from torch.optim import Muon
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers import StoppingCriteria, StoppingCriteriaList
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,   # or load_in_4bit=True
# )
from transformers import get_constant_schedule_with_warmup
from transformers.trainer_utils import set_seed
import torch
import math, re, os, json, gc
from dataclasses import dataclass

# from vllm import LLM, SamplingParams
from sklearn.metrics import accuracy_score
import numpy as np

from typing import Dict, List
import wandb, random
import psutil, GPUtil, time

from transformers import TrainerCallback
from torch.optim import Optimizer
import time, torch, wandb

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids: list[int], device = "cuda"):
        self.stop_ids = torch.tensor(stop_ids, device=device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        # if the last token generated matches any of the stop ids, halt generation
        last_tokens = input_ids[:, -1]
        stopped_mask = torch.isin(last_tokens, self.stop_ids)
        all_done = stopped_mask.all().item()
        return bool(all_done)
class ThroughputCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, temperature, max_new_tokens = 200, top_p = 1.0):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        self.prompts, self.answers = [], []
        for ex in eval_dataset:
            input_ids = ex["input_ids"]
            labels = ex["labels"]
            prompt_tokens = [t for t, l in zip(input_ids, labels) if l == -100]
            prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            self.prompts.append(prompt_text)
            # decode label part to extract the numeric answer
            target_text = tokenizer.decode([t for t in labels if t != -100], skip_special_tokens=True)
            gold_ans = self.extract_number(target_text)
            self.answers.append(gold_ans)
        # Prepare stop tokens
        stop_words = ["Question"]
        stop_ids = set()
        for stop_word in stop_words:
            word_ids = tokenizer.encode(stop_word, add_special_tokens=False)
            stop_ids.update(word_ids)

        stop_ids.add(tokenizer.eos_token_id)
        self.stopping_criteria = StoppingCriteriaList([StopOnTokens(list(stop_ids))])
            
    # Generation processing
    NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
    def extract_number(self, gen_output: str) -> str:
        if "####" in gen_output:
            tail = str(gen_output.split("####")[-1]).strip()
            m = NUM_RE.search(tail)
            if m:
                return m.group(0).strip()
        matches = list(NUM_RE.finditer(gen_output))
        return matches[-1].group(0).strip() if matches else ""
    def normalize_compare(self, gen, ref, eps = 0.1): # it seems like gsm8k dataset only return whole number answer
        def normalize(text: str): #erase ","
            return re.sub(r',', '', text).strip()
        try:
            return abs(float(normalize(gen)) - float(normalize(ref))) < eps
        except Exception:
            return False
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called once after evaluation is finished."""
        model = kwargs["model"]
        model.eval()
        correct, total = 0, 0
        eval_batch_size = 100
        all_prompts, all_preds, all_golds = [], [], []
        for i in range(0, len(self.prompts), eval_batch_size):
            batch_prompts = self.prompts[i:i + eval_batch_size]
            batch_answers = self.answers[i:i + eval_batch_size]

            # Batch tokenize
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=False,
                    stopping_criteria=self.stopping_criteria
                )

            # Decode predictions
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for pred_text, gold_ans in zip(decoded, batch_answers):
                pred_ans = self.extract_number(pred_text)
                print("@@@@ Eval generation: ", pred_text[-1000:])
                print("@@@@ Extracted answer: ", pred_ans)
                print("@@@@ Gold answer: ", gold_ans)
                if self.normalize_compare(pred_ans, gold_ans):
                    correct += 1
                total += 1
                all_prompts.append(pred_text)
                all_golds.append(gold_ans)
                all_preds.append(pred_ans)
        acc = correct / total if total > 0 else 0.0
        wandb.log(
            {"eval/accuracy": acc},
            step=state.global_step,
            commit=False
        )
        eval_table = wandb.Table(columns=["Prompt", "Prediction", "Gold Answer"])
        for p, pred, gold in zip(all_prompts, all_preds, all_golds):
            eval_table.add_data(p, pred, gold)
        wandb.log(
            {"eval/generations": eval_table},
            step=state.global_step
        )
        model.train()
        return control


# Generation processing
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
def extract_number(gen_output: str) -> str:
    if "####" in gen_output:
        tail = str(gen_output.split("####")[-1]).strip()
        m = NUM_RE.search(tail)
        if m:
            return m.group(0).strip()
    matches = list(NUM_RE.finditer(gen_output))
    return matches[-1].group(0).strip() if matches else ""
def extract_mcq(gen_output: str) -> str:
    if "####" in gen_output:
        tail = str(gen_output.split("####")[-1]).strip()
        m = re.search(r"\b([A-E])\b", tail)
        return m.group(1) if m else ""
    return
def normalize_compare(gen, ref, eps = 0.1): # it seems like gsm8k dataset only return whole number answer
    def normalize(text: str): #erase ","
        return re.sub(r',', '', text).strip()
    gen, ref = float(normalize(gen)), float(normalize(ref))
    if abs(gen - ref) < eps:
        return True
    return False
# Load and prepare data
def extract_gsm8k_answer(text : str) -> tuple[str, str]:
    reasoning, final_ans = text.split("####")
    return reasoning.strip(), final_ans.strip()
def gsm8k_prompt_maker(question: str) -> str:
    prompt = f"""
    You are a math solver, think and answer this math reasoning question. Give your reasoning and put your final answer after ####.

    Examples:

    Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
    Answer: Let's think. Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market. #### 18

    Question: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
    Answer: Let's think. It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric. #### 3

    Question: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?
    Answer: Let's think. The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000. #### 70000

    Question: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?
    Answer: Let's think. He sprints 3*3=<<3*3=9>>9 times\nSo he runs 9*60=<<9*60=540>>540 meters. #### 540

    Question: {question}
    Answer: Let's think."""
    return prompt
def aquarat_propmt_maker(question:str, options:str) -> str:
    prompt = f"""
    You are a math solver, think and answer this math reasoning question. Give your reasoning and put your final answer after ####.

    Examples:

    Question: A car is being driven, in a straight line and at a uniform speed, towards the base of a vertical tower. The top of the tower is observed from the car and, in the process, it takes 10 minutes for the angle of elevation to change from 45° to 60°. After how much more time will this car reach the base of the tower?
    Options: ['A)5(√3 + 1)', 'B)6(√3 + √2)', 'C)7(√3 – 1)', 'D)8(√3 – 2)', 'E)None of these']
    Answer: Let's think. Let the height of the building be h. Initially, he was at an angle of 450. tan 45 = h/distance between car and tower. h = distance between car and tower (since tan 45 = 1).\nNow, after 10 minutes, it travelled a certain distance, and angle changed to 600.\ntan 60 = h/x x = h/√3\nSo, in 10 minutes, it has travelled a distance of h – x = h - h/√3.\n10 minutes = h *( 1 – 1√3)\nh can be travelled in 10 / (1 – 1√3).\nTo travel a distance of x, which is h/√3, it takes :\nh = 10 / (1 – 1/√3)\nh / √3 = 10/ √3 * (1 – 1/√3). Multiply numerator and denominator by 1 + √3 ( conjugate of 1 - √3). We get, x = h/√3 = 10 (1 + √3) / 2 = 5* (1 + √3)\nSo, it takes 5(1 + √3) minutes to reach the base of the tower.\nAnswer : A. #### A


    Question:The original price of an item is discounted 22%. A customer buys the item at this discounted price using a $20-off coupon. There is no tax on the item, and this was the only item the customer bought. If the customer paid $1.90 more than half the original price of the item, what was the original price of the item?
    Options: ['A)$61', 'B)$65', 'C)$67.40', 'D)$70', 'E)$78.20']
    Answer: Let's think. Let x be the original price of the item\nDiscounted price = 0.78x\nPayment made by the customer after using the $20 coupon = 0.78x - 20\n0.78x - 20 = x/2 + 1.9\nx = 78.20\nAnswer: E. #### E

    Question: Find out which of the following values is the multiple of X, if it is divisible by 9 and 12?
    Options: ['A)36', 'B)15', 'C)17', 'D)5', 'E)7']
    Answer: Let's think. The number should definitely have these factors 3*3*4\n36 is the number that has these factors\nSo, 36 is the multiple of X\nAnswer is A. #### A

    Question: If the probability that Stock A will increase in value during the next month is 0.56, and the probability that Stock B will increase in value during the next month is 0.74. What is the greatest value for the probability that neither of these two events will occur?
    Options: ['A)0.22', 'B)0.26', 'C)0.37', 'D)0.46', 'E)0.63']
    Answer: Let's think. The probability that stock A does not increase is 0.44, and the probability that stock B does not increase is 0.26. Now, how can the probability that both do not increase be more than individual probability of not increasing for each? So the probability that both do not increase can not be more than 0.26. Basically the probability that both do not increase is between 0 and 0.26. #### B

    Question: {question}
    Options: {options}
    Answer: Let's think."""
    return prompt
def build_sft_data(split, name: str):
    data = []
    if name == "gsm8k":
        for ex in split:
            question, response = ex["question"], ex["answer"]
            reasoning, answer = extract_gsm8k_answer(response)
            prompt = gsm8k_prompt_maker(question)
            if None in [prompt, reasoning, answer]:
                raise ValueError(f"Exists sample with None values: {name}, {i}-th")
            data.append({"prompt" : prompt, "reasoning" : reasoning, "answer" :answer})
    else:
        for i, ex in enumerate(split):
            question, options, rationale, correct = ex["question"], ex["options"], ex["rationale"], ex["correct"]
            prompt = aquarat_propmt_maker(question, options)
            if None in [prompt, rationale, correct]:
                raise ValueError(f"Exists sample with None values: {name}, {i}-th")
            data.append({"prompt" : prompt, "reasoning" : rationale, "answer": correct})
    N = len(data)
    return data[:min(N, 100)] # (prompt, gold reasoning, gold answer) #3205
def load_sft_dataset(name : str):
    if name == "gsm8k":
        gsm8k = load_dataset("openai/gsm8k", "main")
        sft_train = build_sft_data(gsm8k["train"], "gsm8k")
        n_test = len(gsm8k["test"])
        sft_test = random.sample(build_sft_data(gsm8k["test"].select(range(n_test // 2)), "gsm8k"), 100)
        sft_eval = random.sample(build_sft_data(gsm8k["test"].select(range(n_test// 2, n_test)), "gsm8k"), 100)
    else: #aqua_rat
        aqua_rat = load_dataset("deepmind/aqua_rat", "raw")
        sft_train = build_sft_data(aqua_rat["train"], "aqua")
        sft_test = random.sample(build_sft_data(aqua_rat["test"], "aqua"), 100)
        sft_eval = random.sample(build_sft_data(aqua_rat["validation"], "aqua"), 100)
    return sft_train, sft_test, sft_eval

def tokenize_sft_data(data, tokenizer, max_length = 4096):
    tokenized = [] # input_ids, attention_mask, labels
    for ex in data :
        question, reasoning, answer = ex["prompt"], ex["reasoning"], ex["answer"]
        p_ids = tokenizer(question, add_special_tokens = False)["input_ids"]
        t_str = reasoning.strip() + " #### " + answer.strip() + tokenizer.eos_token
        t_ids = tokenizer(t_str, add_special_tokens = False)["input_ids"]

        total_len = len(p_ids) + len(t_ids)
        if total_len > max_length:
            overflow = total_len - max_length
            raise ValueError(f"Max Length Overflow {overflow}")

        input_ids = p_ids + t_ids # combine tokens duoc, but combine with a list needs a twist  (below)
        attention_mask = [1]*len(input_ids)
        labels = [-100] * len(p_ids) + t_ids[:] #In PyTorch’s CrossEntropyLoss, the default ignore_index is -100. So loss is only computed on labels[i] != -100
        tokenized.append({"input_ids" : input_ids, "attention_mask" : attention_mask, "labels" : labels})

    return tokenized


@dataclass #trainer will call this class everytime it forms a mini batch -> pad lengths of samples in the minibatch to be equal (would it affect packing?)
class ChatLMDataCollator:
    tokenizer: AutoTokenizer
    pad_to_multiply_of: int = 8
    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiply_of:
            max_len = int(math.ceil(max_len / self.pad_to_multiply_of)*self.pad_to_multiply_of)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id]*pad_len)
            attention_mask.append(f["attention_mask"] + [0]*pad_len) # 0 is for dont look
            labels.append(f["labels"] + [-100]*pad_len)
        try:
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
        except Exception as e:
            print("❌ Tensor conversion failed:")
            print("Error type:", type(e).__name__)
            print("Message:", e)
            print("input_ids sample:", input_ids[:2] if isinstance(input_ids, list) else "N/A")
            print("attention_mask sample:", attention_mask[:2] if isinstance(attention_mask, list) else "N/A")
            print("labels sample:", labels[:2] if isinstance(labels, list) else "N/A")
            raise

        outputs = {
            "input_ids": torch.tensor(input_ids, dtype = torch.long), "attention_mask" : torch.tensor(attention_mask, dtype = torch.long), "labels" : torch.tensor(labels, dtype = torch.long)
        }
        return outputs

# Load model
def load_model(model_or_ckpt : str, tokenizer : AutoTokenizer):
    #tokenizer = AutoTokenizer.from_pretrained(model_or_ckpt)
    model = AutoModelForCausalLM.from_pretrained(model_or_ckpt, torch_dtype = torch.bfloat16, device_map="balanced")
    model.gradient_checkpointing_enable()
    if tokenizer.pad_token is None: # tokenizer is initialized during vllm | But vllm's tokenizer's setting is different and cannot be loaded
        tokenizer.pad_token = tokenizer.eos_token
        # ensure model + config are consistent
    model.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model
class MultiOptim(Optimizer):
    """
    Combine multiple torch optimizers into one optimizer-compatible object
    with proper .param_groups for HF schedulers.
    """
    def __init__(self, optimizers):
        # Base init requires a dummy param group
        object.__setattr__(self, "_init_in_progress", True)
        super().__init__([{"params": []}], defaults={})
        object.__setattr__(self, "_init_in_progress", False)
        self.optimizers = optimizers

        # Merge param_groups (shared references so schedulers update both)
        self.param_groups = []
        for opt in optimizers:
            self.param_groups.extend(opt.param_groups)

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for opt in self.optimizers:
            if closure is not None:
                closure()
            opt.step()

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)

    def add_param_group(self, param_group):
        if getattr(self, "_init_in_progress", False):
            return super().add_param_group(param_group)
        # Prevent accidental dynamic additions later
        raise NotImplementedError("MultiOptim does not support dynamic param groups.")

def build_muon(model, lr, muon_weight_decay=0.0,
               adamw_weight_decay=0.01):
    def split_params_for_muon_adamw(model, muon_params, adamw_params):
        seen = set()

        def add(dst, p):
            pid = id(p)
            if pid not in seen and p.requires_grad:
                seen.add(pid)
                dst.append(p)

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            lname = name.lower()

            is_norm = ("norm" in lname) or ("layernorm" in lname) or ("rms" in lname)
            is_embed = ("embed" in lname)            # catches embed_tokens & tied weights
            is_head  = ("lm_head" in lname)
            is_rope  = ("rotary" in lname) or ("rope" in lname) or ("pos" in lname)
            is_bias  = lname.endswith(".bias") # gpt says LLama3.1 doesn't have bias (bias=False)

            if (p.ndim == 2
                and not (is_embed or is_head or is_norm or is_rope)):
                # true projection matrices (attn & MLP)
                add(muon_params, p)
            else:
                # embeddings, lm_head, norms, biases, others
                add(adamw_params, p)

        return muon_params, adamw_params
    
    muon_params, adamw_params = split_params_for_muon_adamw(model, [],[])  
    optimizers = []
    if muon_params:
        optimizers.append(
            torch.optim.Muon(
                muon_params,
                lr=lr,
                weight_decay=muon_weight_decay,
                adjust_lr_fn = "match_rms_adamw",
            )
        )
    if adamw_params:
        optimizers.append(
            torch.optim.AdamW(
                adamw_params,
                lr=lr,
                weight_decay=adamw_weight_decay,
                eps= 1e-8,
            )
        )

    return optimizers[0] if len(optimizers) == 1 else MultiOptim(optimizers)


def build_optimizer(
    model,
    optimizer_name: str,
    lr: float,
    adamw_weight_decay: float,
    adamw_eps: float,
    muon_weight_decay: float,
    muon_momentum: float,
    muon_eps: float,
    muon_nesterov: bool = False,
    muon_ns_coeff: float = None,
    muon_ns_steps: int = None
):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),lr=lr,betas=(0.9, 0.999),eps=adamw_eps, weight_decay=adamw_weight_decay,
        )
    elif optimizer_name == "muon":
        return build_muon(
            model,lr,muon_weight_decay=muon_weight_decay,adamw_weight_decay=adamw_weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
def build_trainer(tokenizer, model, train, eval, warmup_ratio, epochs, batch_size, grad_accum, max_new_tokens, temperature, top_p,
                  output_dir, log_every, seed,
                  optimizer_name : str, lr :float, adamw_weight_decay :float, adamw_eps : float, #adamw
                  muon_weight_decay: float, muon_eps :float, muon_momentum :float, muon_nesterov = True, muon_ns_coeff = (3.4445, -4.775, 2.0315), muon_ns_steps  = 5): #muon
    steps_per_epoch = math.ceil(len(train) / batch_size)
    updates_per_epoch = math.ceil(steps_per_epoch / max(1, grad_accum))
    total_steps = max(1, updates_per_epoch * epochs) #accum over all epochs, account for grad accum
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    print("### Total Steps: ", total_steps)
    print("### Warm up Steps: ", warmup_steps)
    args = TrainingArguments(
        output_dir = output_dir,
        num_train_epochs = epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        gradient_accumulation_steps = grad_accum,
        # learning_rate=lr,
        # lr_scheduler_type = "constant_with_warmup",
        # warmup_steps = warmup_steps,
        #warmup_ratio = warmup_ratio,
        logging_steps = log_every,
        logging_strategy = "steps",
        seed = seed,
        save_steps = log_every,
        eval_steps = log_every,
        eval_strategy = "steps",
        save_strategy = "steps",
        eval_accumulation_steps = 2,
        save_total_limit = 8,
        bf16=True,
        report_to = ["wandb"],
        load_best_model_at_end = True
    ) #don't need to intialize lr, optimizer will do that

    data_collator = ChatLMDataCollator(tokenizer = tokenizer, pad_to_multiply_of=8)

    optimizer = build_optimizer(model, optimizer_name, lr, adamw_weight_decay, adamw_eps,
         muon_weight_decay, muon_momentum=muon_momentum, muon_eps=muon_eps, muon_nesterov=muon_nesterov, muon_ns_coeff= muon_ns_coeff, muon_ns_steps=muon_ns_steps)

    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

    trainer = Trainer(model = model, args = args, train_dataset = train, eval_dataset = eval, data_collator = data_collator)
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    gen_callback = ThroughputCallback(tokenizer = tokenizer, eval_dataset=eval, temperature = temperature, max_new_tokens=max_new_tokens, top_p= top_p)
    trainer.add_callback(gen_callback)
    return trainer

def main():
    model_name = "meta-llama/Llama-3.2-1B"
    optimizer_name = "muon" 
    save_dir = "./demo_muon"
    lr, weight_decay = [0.00002, 0.001], [0.05, 0.1] # adamw , muon
    eps = [1e-8, 1e-7]
    muon_momentum = 0.1
    warmup_ratio, epochs, batch_size, grad_accum = 0.1, 1, 2, 4
    log_every, seed = 2, 42
    gmu = 0.7
    max_new_tokens = 512
    temperature = 0.2 
    top_p = 1.0
    set_seed(seed)
    random.seed(seed)

    #load tokenizer (setting pad_token inside load_vllm)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    stage1_dir = os.path.join(save_dir, optimizer_name + "_demo_muon")
    
    train_gsm8k, test_gsm8k, eval_gsm8k = load_sft_dataset("gsm8k")
    sft_train_gsm8k = tokenize_sft_data(train_gsm8k, tokenizer)
    sft_eval_gsm8k = tokenize_sft_data(eval_gsm8k, tokenizer)
    

    hf_model = load_model(model_name, tokenizer)
    print("After loading & Before training: ", hf_model.hf_device_map)
    trainer1 = build_trainer(tokenizer,  hf_model, sft_train_gsm8k, sft_eval_gsm8k, warmup_ratio, epochs, batch_size, grad_accum, max_new_tokens, temperature, top_p,
                    stage1_dir, log_every, seed,
                    optimizer_name, lr[0], weight_decay[0], eps[0],
                    weight_decay[1], eps[1], muon_momentum)
    print("before training")
    trainer1.train()
    print("after training")

if __name__ == "__main__":
    main()