import os
import sys

# Add csm-mlx directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "csm-mlx"))

import torch
import torchaudio
import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download

# Import both PyTorch and MLX implementations
from generator import load_csm_1b as load_csm_1b_torch, Segment as TorchSegment
from csm_mlx import CSM as MLXCSM, csm_1b as csm_1b_mlx, Segment as MLXSegment
import mlx.core as mx

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

# Default prompts are available at https://hf.co/sesame/csm-1b
prompt_filepath_conversational_a = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_a.wav"
)
prompt_filepath_conversational_b = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_b.wav"
)

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam"
        ),
        "audio": prompt_filepath_conversational_a
    },
    "conversational_b": {
        "text": (
            "like a super Mario level. Like it's very like high detail. And like, once you get "
            "into the park, it just like, everything looks like a computer game and they have all "
            "these, like, you know, if, if there's like a, you know, like in a Mario game"
        ),
        "audio": prompt_filepath_conversational_b
    }
}

SPACE_INTRO_TEXT = """\
# Sesame CSM 1B Demo

Generate conversations using CSM 1B (Conversational Speech Model). 
Each line in the conversation will alternate between Speaker A and B.
"""

DEFAULT_CONVERSATION = """\
Hey how are you doing.
Pretty good, pretty good.
I'm great, so happy to be speaking to you.
Me too, this is some cool stuff huh?
"""

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor | mx.array:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    # Resample is lazy so we can always call it
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int, backend: str) -> TorchSegment | MLXSegment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    
    if backend == "mlx":
        # Convert torch tensor to MLX array
        if isinstance(audio_tensor, torch.Tensor):
            audio_tensor = mx.array(audio_tensor.numpy())
        return MLXSegment(text=text, speaker=speaker, audio=audio_tensor)
    else:
        # Convert MLX array to torch tensor if needed
        if isinstance(audio_tensor, mx.array):
            audio_tensor = torch.from_numpy(audio_tensor.numpy())
        return TorchSegment(text=text, speaker=speaker, audio=audio_tensor)

def get_backend():
    """Automatically select the best available backend"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():  # Check for MPS (Apple Silicon)
        return "mlx"
    else:
        return "cpu"

def infer(
    text_prompt_speaker_a,
    text_prompt_speaker_b,
    audio_prompt_speaker_a,
    audio_prompt_speaker_b,
    conversation_input,
) -> tuple[int, np.ndarray]:
    backend = get_backend()
    print(f"Using backend: {backend}")

    try:
        if backend == "mlx":
            # Use csm-mlx's implementation
            from mlx_lm.sample_utils import make_sampler
            from csm_mlx import generate as mlx_generate
            
            # Initialize model
            generator = MLXCSM(csm_1b_mlx())
            weight = hf_hub_download(repo_id="senstella/csm-1b-mlx", filename="ckpt.safetensors")
            generator.load_weights(weight)
            sample_rate = 24000

            # Prepare prompts
            prompt_a = prepare_prompt(text_prompt_speaker_a, 0, audio_prompt_speaker_a, sample_rate, backend)
            prompt_b = prepare_prompt(text_prompt_speaker_b, 1, audio_prompt_speaker_b, sample_rate, backend)
            context = [prompt_a, prompt_b]

            # Generate conversation
            generated_segments = []
            conversation_lines = [line.strip() for line in conversation_input.strip().split("\n") if line.strip()]
            
            for i, line in enumerate(conversation_lines):
                speaker_id = i % 2
                audio = mlx_generate(
                    generator,
                    text=line,
                    speaker=speaker_id,
                    context=context + generated_segments,
                    max_audio_length_ms=10_000,
                    sampler=make_sampler(temp=0.8, top_k=50)
                )
                generated_segments.append(MLXSegment(text=line, speaker=speaker_id, audio=audio))

            # Concatenate all generations
            all_audio = mx.concat([seg.audio for seg in generated_segments], axis=0)
            # Convert to 16-bit PCM format that Gradio expects
            audio_array = (all_audio * 32768).astype(mx.int16)
            # Convert to numpy array
            audio_array = np.array(audio_array.tolist(), dtype=np.int16)

        else:
            # Original PyTorch implementation
            generator = load_csm_1b_torch(backend)
            sample_rate = generator.sample_rate

            # Prepare prompts
            prompt_a = prepare_prompt(text_prompt_speaker_a, 0, audio_prompt_speaker_a, sample_rate, backend)
            prompt_b = prepare_prompt(text_prompt_speaker_b, 1, audio_prompt_speaker_b, sample_rate, backend)
            prompt_segments = [prompt_a, prompt_b]
            
            generated_segments = []
            conversation_lines = [line.strip() for line in conversation_input.strip().split("\n") if line.strip()]
            
            for i, line in enumerate(conversation_lines):
                speaker_id = i % 2
                audio_tensor = generator.generate(
                    text=line,
                    speaker=speaker_id,
                    context=prompt_segments + generated_segments,
                    max_audio_length_ms=10_000,
                )
                generated_segments.append(TorchSegment(text=line, speaker=speaker_id, audio=audio_tensor))

            all_audio = torch.cat([seg.audio for seg in generated_segments], dim=0)
            audio_array = (all_audio * 32768).to(torch.int16).cpu().numpy()

        return sample_rate, audio_array

    except Exception as e:
        raise gr.Error(f"Error generating audio: {e}")

def update_prompt(speaker):
    if speaker in SPEAKER_PROMPTS:
        return (
            SPEAKER_PROMPTS[speaker]["text"],
            SPEAKER_PROMPTS[speaker]["audio"]
        )
    return None, None

def create_speaker_prompt_ui(speaker_name: str):
    speaker_dropdown = gr.Dropdown(
        choices=list(SPEAKER_PROMPTS.keys()),
        label="Select a predefined speaker",
        value=speaker_name
    )
    with gr.Accordion("Or add your own voice prompt", open=False):
        text_prompt_speaker = gr.Textbox(
            label="Speaker prompt",
            lines=4,
            value=SPEAKER_PROMPTS[speaker_name]["text"]
        )
        audio_prompt_speaker = gr.Audio(
            label="Speaker prompt",
            type="filepath",
            value=SPEAKER_PROMPTS[speaker_name]["audio"]
        )

    return speaker_dropdown, text_prompt_speaker, audio_prompt_speaker

with gr.Blocks() as app:
    gr.Markdown(SPACE_INTRO_TEXT)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Speaker A")
            speaker_a_dropdown, text_prompt_speaker_a, audio_prompt_speaker_a = create_speaker_prompt_ui(
                "conversational_a"
            )

        with gr.Column():
            gr.Markdown("### Speaker B")
            speaker_b_dropdown, text_prompt_speaker_b, audio_prompt_speaker_b = create_speaker_prompt_ui(
                "conversational_b"
            )

    # Update prompts when dropdown changes
    speaker_a_dropdown.change(
        fn=update_prompt,
        inputs=[speaker_a_dropdown],
        outputs=[text_prompt_speaker_a, audio_prompt_speaker_a]
    )
    speaker_b_dropdown.change(
        fn=update_prompt,
        inputs=[speaker_b_dropdown],
        outputs=[text_prompt_speaker_b, audio_prompt_speaker_b]
    )

    gr.Markdown("## Conversation")
    conversation_input = gr.TextArea(
        label="Enter conversation (alternating lines between speakers)",
        lines=10,
        value=DEFAULT_CONVERSATION
    )
    
    generate_btn = gr.Button("Generate conversation", variant="primary")
    audio_output = gr.Audio(label="Generated conversation")

    generate_btn.click(
        fn=infer,
        inputs=[
            text_prompt_speaker_a,
            text_prompt_speaker_b,
            audio_prompt_speaker_a,
            audio_prompt_speaker_b,
            conversation_input,
        ],
        outputs=[audio_output],
    )

if __name__ == "__main__":
    app.launch()
