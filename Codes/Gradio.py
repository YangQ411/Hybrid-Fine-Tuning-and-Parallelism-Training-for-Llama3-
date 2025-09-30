import torch
import re
import gradio as gr
from transformers import AutoTokenizer
from config import get_lora_config, get_ds_config
from model import get_fine_tune_model


print("Loading model with LoRA weights...")

lora_config = get_lora_config()
ds_config = get_ds_config()


model = get_fine_tune_model(ds_config, lora_config)
tokenizer = AutoTokenizer.from_pretrained(ds_config["model_name"])
tokenizer.pad_token = tokenizer.eos_token


ckpt = torch.load("best_fine_tune_LoRA_llama3_bf16.pt", map_location="cpu")
model.load_state_dict(ckpt["lora_state_dict"], strict=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(torch.bfloat16).to(device)
model.eval()
print("Model ready for inference.")


def respond(message, history, system_message, max_tokens, temperature, top_p):
    prompt = f"{system_message.strip()}\nquestion: {message.strip()}\nresponse:"
    print("=== PROMPT ===")
    print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        num_beams=1
    )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=== OUTPUT ===")
    print(full_output)

    if "response:" in full_output:
        answer = full_output.split("response:", 1)[-1].strip()
    else:
        answer = full_output.strip()

    for stop_token in ["question:", "#", "\n\n", "##", "Chapter", "def ", "print("]:
        if stop_token in answer:
            answer = answer.split(stop_token, 1)[0].strip()

    return answer if answer else "Sorry, I haven't found the answer yet."


demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value="", label="(Optional) System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=1.0, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, label="Top-p (nucleus sampling)")
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)
