"""Forced alignment (transcript -> word timestamps) with torchaudio's MMS_FA CTC aligner.

Needs: torch + torchaudio (see requirements.txt). The model (~1.2 GB) is downloaded on first use.
"""
from .io import load_audio
from .text import normalize_words


def _load_model(bundle, device="cpu"):
    import torch
    from torchaudio.pipelines._wav2vec2 import utils

    # Load weights with weights_only=True to prevent pickle overhead and access violations
    cached = torch.hub.load_state_dict_from_url(bundle._path, map_location="cpu", weights_only=True)
    if getattr(bundle, "_remove_aux_axis", None):
        utils._remove_aux_axes(cached, bundle._remove_aux_axis)
    with torch.device("meta"):
        meta_model = utils._get_model(bundle._model_type, bundle._params)
    meta_model.load_state_dict(cached, assign=True)
    model = utils._extend_model(meta_model, normalize_waveform=bundle._normalize_waveform,
                                apply_log_softmax=True, append_star=False)
    model.eval()
    return model.to(device)


def align_words(audio_path, transcript: str, device: str = "cpu"):
    import torch
    import torchaudio

    words = normalize_words(transcript)
    if not words:
        raise ValueError("Transcript has no alignable words")
    bundle = torchaudio.pipelines.MMS_FA
    model = _load_model(bundle, device=device)
    tokenizer, aligner = bundle.get_tokenizer(), bundle.get_aligner()
    y = load_audio(audio_path, sr=int(bundle.sample_rate))
    wav = torch.from_numpy(y)[None].to(device)
    with torch.inference_mode():
        emission, _ = model(wav)
    spans = aligner(emission[0], tokenizer(words))
    ratio = wav.size(1) / emission.size(1) / bundle.sample_rate
    return [{"word": w, "start": float(sp[0].start * ratio), "end": float(sp[-1].end * ratio)}
            for w, sp in zip(words, spans)]
