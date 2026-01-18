# Installation:
# pip install torch transformers

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Determine device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Add the EOS token as PAD token to avoid warnings
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", pad_token_id=tokenizer.eos_token_id
).to(torch_device)

def generate_text(prompt: str, max_new_tokens: int = 50, temperature: float = 0.7) -> str:
    """Generate text continuation for a given prompt.

    Args:
        prompt (str): Input text prompt.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
    Returns:
        str: Generated text (prompt + continuation).
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(torch_device)
    # Generate output ids
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode to string
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    demo_prompt = "Once upon a time"
    generated = generate_text(demo_prompt)
    print("Prompt:", demo_prompt)
    print("Generated:", generated)
