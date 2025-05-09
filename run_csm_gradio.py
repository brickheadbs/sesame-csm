import os
import sys
from typing import Union, Any  # Add this import for type hints
import time  # Add this import
import psutil  # Add this import

# Add csm-mlx directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "csm-mlx"))

import torch
import torchaudio
import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download

# Import both PyTorch and MLX implementations
from generator import load_csm_1b as load_csm_1b_torch, Segment as TorchSegment

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

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> Any:  # Changed return type
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    # Resample is lazy so we can always call it
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int, backend: str) -> Any:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)

    if backend == "mlx":
        # Only import and use MLX-related code when using MLX backend
        global mx, MLXSegment
        from csm_mlx import Segment as MLXSegment
        import mlx.core as mx

        # Convert torch tensor to MLX array
        if isinstance(audio_tensor, torch.Tensor):
            audio_tensor = mx.array(audio_tensor.numpy())
        return MLXSegment(text=text, speaker=speaker, audio=audio_tensor)
    else:
        # For non-MLX backends, we know it's always a torch tensor
        return TorchSegment(text=text, speaker=speaker, audio=audio_tensor)

def setup_mlx():
    """Setup MLX imports and return True if successful"""
    try:
        global MLXCSM, csm_1b_mlx, MLXSegment, mx
        from csm_mlx import CSM as MLXCSM, csm_1b as csm_1b_mlx, Segment as MLXSegment
        import mlx.core as mx
        return True
    except ImportError:
        return False

def get_backend():
    """Automatically select the best available backend"""
    # Allow override via environment variable
    forced_backend = os.environ.get("CSM_BACKEND", "").lower()
    if forced_backend in ["cpu", "cuda", "mlx"]:
        if forced_backend == "mlx" and setup_mlx():
            return "mlx"
        elif forced_backend in ["cpu", "cuda"]:
            return forced_backend

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():  # Check for MPS (Apple Silicon)
        if setup_mlx():
            return "mlx"
    return "cpu"

def get_memory_usage(backend: str) -> float:
    """Get peak memory usage based on backend"""
    if backend == "mlx":
        return mx.get_peak_memory() / 1024**3  # Updated from mx.metal.get_peak_memory()
    elif backend == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
    elif backend == "cpu":
        # Get process memory usage
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**3  # Convert bytes to GB
    return 0

def reset_memory_tracking(backend: str):
    """Reset memory tracking based on backend"""
    if backend == "mlx":
        mx.reset_peak_memory()
    elif backend == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    elif backend == "cpu":
        # For CPU, we can trigger garbage collection
        import gc
        gc.collect()

def infer(
    text_prompt_speaker_a,
    text_prompt_speaker_b,
    audio_prompt_speaker_a,
    audio_prompt_speaker_b,
    conversation_input,
) -> tuple[Any]:
    backend = get_backend()
    print(f"Using backend: {backend}")

    try:
        # Track timing and memory
        start_total = time.time()
        reset_memory_tracking(backend)

        # Track text encoding/model loading time
        start_load = time.time()
        if backend == "mlx":
            # MLX setup code
            from mlx_lm.sample_utils import make_sampler
            from csm_mlx import generate as mlx_generate

            generator = MLXCSM(csm_1b_mlx())
            weight = hf_hub_download(repo_id="senstella/csm-1b-mlx", filename="ckpt.safetensors")
            generator.load_weights(weight)
            sample_rate = 24000
        else:
            generator = load_csm_1b_torch(backend)
            sample_rate = generator.sample_rate

        load_time = time.time() - start_load
        load_mem = get_memory_usage(backend)

        # Track generation time
        start_gen = time.time()
        reset_memory_tracking(backend)

        # Prepare prompts
        prompt_a = prepare_prompt(text_prompt_speaker_a, 0, audio_prompt_speaker_a, sample_rate, backend)
        prompt_b = prepare_prompt(text_prompt_speaker_b, 1, audio_prompt_speaker_b, sample_rate, backend)
        context = [prompt_a, prompt_b]

        # Generate conversation
        generated_segments = []
        conversation_lines = [line.strip() for line in conversation_input.strip().split("\n") if line.strip()]

        for i, line in enumerate(conversation_lines):
            speaker_id = i % 2
            if backend == "mlx":
                audio = mlx_generate(
                    generator,
                    text=line,
                    speaker=speaker_id,
                    context=context + generated_segments,
                    max_audio_length_ms=10_000,
                    sampler=make_sampler(temp=0.8, top_k=50)
                )
                generated_segments.append(MLXSegment(text=line, speaker=speaker_id, audio=audio))
            else:
                # PyTorch (CPU/CUDA) generation
                audio_tensor = generator.generate(
                    text=line,
                    speaker=speaker_id,
                    context=context + generated_segments,
                    max_audio_length_ms=10_000,
                )
                generated_segments.append(TorchSegment(text=line, speaker=speaker_id, audio=audio_tensor))

        # Concatenate all generations
        if backend == "mlx":
            all_audio = mx.concat([seg.audio for seg in generated_segments], axis=0)
            audio_array = (all_audio * 32768).astype(mx.int16)
            audio_array = np.array(audio_array.tolist(), dtype=np.int16)
        else:
            all_audio = torch.cat([seg.audio for seg in generated_segments], dim=0)
            audio_array = (all_audio * 32768).to(torch.int16).cpu().numpy()

        gen_time = time.time() - start_gen
        gen_mem = get_memory_usage(backend)

        # Format stats with memory info for all backends
        stats = {
            "load_time": f"**Model Loading Time:** {load_time:.2f}s",
            "gen_time": f"**Generation Time:** {gen_time:.2f}s",
            "total_time": f"**Total Time:** {time.time() - start_total:.2f}s",
            "load_mem": f"**Model Loading Memory:** {load_mem:.2f}GB" if load_mem > 0 else "**Model Loading Memory:** N/A",
            "gen_mem": f"**Generation Memory:** {gen_mem:.2f}GB" if gen_mem > 0 else "**Generation Memory:** N/A",
            "total_mem": f"**Total Peak Memory:** {max(load_mem, gen_mem):.2f}GB" if max(load_mem, gen_mem) > 0 else "**Total Peak Memory:** N/A",
        }

        # Return tuple of (audio_tuple, stats...)
        return (sample_rate, audio_array), stats["load_time"], stats["gen_time"], stats["total_time"], stats["load_mem"], stats["gen_mem"], stats["total_mem"]

    except Exception as e:
        error_msg = f"Error generating audio: {e}"
        empty_stats = ["N/A"] * 6  # 6 stats fields
        raise gr.Error(error_msg)

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

    # Add Stats Box
    with gr.Group(visible=True) as stats_group:
        gr.Markdown("### 🔍 Generation Stats")
        with gr.Row():
            with gr.Column(scale=1):
                load_mem = gr.Markdown("**Model Loading Memory:** N/A")
                gen_mem = gr.Markdown("**Generation Memory:** N/A")
                total_mem = gr.Markdown("**Total Peak Memory:** N/A")
            with gr.Column(scale=1):
                load_time = gr.Markdown("**Model Loading Time:** N/A")
                gen_time = gr.Markdown("**Generation Time:** N/A")
                total_time = gr.Markdown("**Total Time:** N/A")

    generate_btn.click(
        fn=infer,
        inputs=[
            text_prompt_speaker_a,
            text_prompt_speaker_b,
            audio_prompt_speaker_a,
            audio_prompt_speaker_b,
            conversation_input,
        ],
        outputs=[
            audio_output,
            load_time,
            gen_time,
            total_time,
            load_mem,
            gen_mem,
            total_mem,
        ],
    )

if __name__ == "__main__":
    # Launch the app and automatically open in browser
    app.launch(
        share=False,  # Don't create public URL
        inbrowser=True,  # Automatically open in default browser
        server_name="0.0.0.0",  # Listen on all network interfaces
        server_port=7860,  # Default Gradio port
        show_error=True,  # Show detailed error messages
    )
