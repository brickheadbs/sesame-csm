import os
import sys
from typing import Any

# Add csm-mlx directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "csm-mlx"))

import torch
import torchaudio
import numpy as np
from huggingface_hub import hf_hub_download

# Import PyTorch implementation
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

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> Any:
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
        global MLXCSM, csm_1b_mlx, MLXSegment, mx, mlx_generate
        from csm_mlx import CSM as MLXCSM, csm_1b as csm_1b_mlx, Segment as MLXSegment, generate as mlx_generate
        import mlx.core as mx
        return "mlx"
    else:
        return "cpu"

def main():
    backend = get_backend()
    print(f"Using backend: {backend}")

    # Initialize model based on backend
    if backend == "mlx":
        from mlx_lm.sample_utils import make_sampler

        # Initialize MLX model
        generator = MLXCSM(csm_1b_mlx())
        weight = hf_hub_download(repo_id="senstella/csm-1b-mlx", filename="ckpt.safetensors")
        generator.load_weights(weight)
        sample_rate = 24000
    else:
        # Initialize PyTorch model
        generator = load_csm_1b_torch(backend)
        sample_rate = generator.sample_rate

    # Prepare prompts
    prompt_a = prepare_prompt(
        SPEAKER_PROMPTS["conversational_a"]["text"],
        0,
        SPEAKER_PROMPTS["conversational_a"]["audio"],
        sample_rate,
        backend
    )

    prompt_b = prepare_prompt(
        SPEAKER_PROMPTS["conversational_b"]["text"],
        1,
        SPEAKER_PROMPTS["conversational_b"]["audio"],
        sample_rate,
        backend
    )

    # Generate conversation
    conversation = [
        {"text": "Hey how are you doing?", "speaker_id": 0},
        {"text": "Pretty good, pretty good. How about you?", "speaker_id": 1},
        {"text": "I'm great! So happy to be speaking with you today.", "speaker_id": 0},
        {"text": "Me too! This is some cool stuff, isn't it?", "speaker_id": 1}
    ]

    # Generate each utterance
    generated_segments = []
    context = [prompt_a, prompt_b] if backend == "mlx" else [prompt_a, prompt_b]

    for utterance in conversation:
        print(f"Generating: {utterance['text']}")

        if backend == "mlx":
            audio = mlx_generate(
                generator,
                text=utterance['text'],
                speaker=utterance['speaker_id'],
                context=context + generated_segments,
                max_audio_length_ms=10_000,
                sampler=make_sampler(temp=0.8, top_k=50)
            )
            generated_segments.append(MLXSegment(
                text=utterance['text'], 
                speaker=utterance['speaker_id'], 
                audio=audio
            ))
        else:
            audio_tensor = generator.generate(
                text=utterance['text'],
                speaker=utterance['speaker_id'],
                context=context + generated_segments,
                max_audio_length_ms=10_000,
            )
            generated_segments.append(TorchSegment(
                text=utterance['text'], 
                speaker=utterance['speaker_id'], 
                audio=audio_tensor
            ))

    # Concatenate all generations and save
    if backend == "mlx":
        all_audio = mx.concat([seg.audio for seg in generated_segments], axis=0)
        # Convert to numpy array with proper scaling
        audio_array = (all_audio * 32768).astype(mx.int16)
        audio_array = np.array(audio_array.tolist(), dtype=np.int16)
        # Save using torchaudio
        torchaudio.save(
            "full_conversation.wav",
            torch.from_numpy(audio_array).unsqueeze(0),
            sample_rate
        )
    else:
        all_audio = torch.cat([seg.audio for seg in generated_segments], dim=0)
        torchaudio.save(
            "full_conversation.wav",
            all_audio.unsqueeze(0).cpu(),
            sample_rate
        )

    print("Successfully generated full_conversation.wav")

if __name__ == "__main__":
    main() 