# Small Companion AI (~800 MB)
small_ai_model_name = "bigscience/bloom-560m"  # ~800 MB example, lightweight
small_ai_tokenizer = AutoTokenizer.from_pretrained(small_ai_model_name)
small_ai_model = AutoModelForCausalLM.from_pretrained(
    small_ai_model_name, torch_dtype=torch.float16, device_map="auto"
)
small_ai_pipe = pipeline("text-generation", model=small_ai_model, tokenizer=small_ai_tokenizer)

# -------------------------------
# Update AI handler
# -------------------------------
def run_ai(ai_type, msg):
    # Combine context for all file-based models
    prompt = context_text + "\n\nUser Query: " + msg
    now = datetime.now().time()

    if ai_type == "small_ai":
        # Companion AI responds using files and chat context
        result = small_ai_pipe(prompt, max_length=256, do_sample=True)[0]['generated_text']
        return result

    # existing handlers for other models
    elif ai_type == "starcoder":
        return starcoder_pipe(prompt, max_length=256, do_sample=True)[0]['generated_text']
    elif ai_type == "codeparrot":
        return codeparrot_pipe(prompt, max_length=128, do_sample=True)[0]['generated_text']
    elif ai_type == "qwen":
        if (now >= qwen_start) or (now <= qwen_end):
            return qwen_pipe(prompt, max_length=128, do_sample=True)[0]['generated_text']
        else:
            return "Qwen2.5-Coder-0.5B available 11 PM – 6:45 AM."
    elif ai_type == "qwen_large":
        return qwen_large_pipe(prompt, max_length=256, do_sample=True)[0]['generated_text']
    elif ai_type == "deepseek":
        return deepseek_pipe(prompt, max_length=256, do_sample=True)[0]['generated_text']
    elif ai_type == "qwen_7b":
        return qwen_7b_pipe(prompt, max_length=256, do_sample=True)[0]['generated_text']
    elif ai_type == "heavy":
        return heavy_pipe(prompt, max_length=256, do_sample=True)[0]['generated_text']
    else:
        return "Unknown AI type."