# Silero VAD ONNX asset

`silero_vad_v6.onnx` is the sequence-form Silero VAD v6 export distributed by faster-whisper. It evaluates many 512-sample frames per ONNX Runtime call while preserving Silero's recurrent state and timestamp rules.

- Source revision: `SYSTRAN/faster-whisper@ed9a06cd89a93e47838f564998a6c09b655d7f43`
- Source path: `faster_whisper/assets/silero_vad_v6.onnx`
- Runtime reference: `faster_whisper/vad.py`
- SHA-256: `914fd98ac0a73d69ba1e70c9b1d66acb740eff90500dfde08b89a961b168a6a9`
- Upstream model: <https://github.com/snakers4/silero-vad>
- Distributor: <https://github.com/SYSTRAN/faster-whisper>

The asset, Ruby timestamp state machine, and sequence-runner behavior are MIT-licensed. `LICENSE.silero-vad` covers the model and upstream timestamp behavior, while `LICENSE.faster-whisper` covers the repository distributing this exact sequence export and its integration behavior. Both notices accompany the packaged code and asset.

The Ruby runner deliberately avoids two boundary errors: exactly divisible waveforms do not gain an extra frame, and constructing a first row's zero context never mutates the preceding audio frame. Direct sequence ONNX/JIT compatibility uses bounded 256-frame calls. The packed `auto`/`torch` compatibility path instead honors the validated `vad_block_frames` ceiling exactly while carrying recurrent state between calls.
