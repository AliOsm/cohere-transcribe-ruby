# Word-alignment attribution

## Downloaded MMS model

The optional word-alignment runtime downloads the full-precision ONNX export of `MahmoudAshraf/mms-300m-1130-forced-aligner` into the user's Hugging Face cache. The export is pinned to `onnx-community/mms-300m-1130-forced-aligner-ONNX@2100fb247d8e43962eef24491597fbeb8b469531` and is derived from the source checkpoint revision `49402e9577b1158620820667c218cd494cc44486`.

The model is licensed under CC-BY-NC-4.0, including its non-commercial-use restriction. The model weights are downloaded only when word mode is used and are not distributed inside this gem. The runtime pins and verifies these artifacts before loading them:

- `onnx/model.onnx`: 1,262,529,881 bytes, SHA-256 `429e5d05c62acc8a9264db874a1b131e359fc626e40c253ac7b1fe52b11149b4`
- `onnx/model_fp16.onnx`: 631,591,191 bytes, SHA-256 `e98082b382375f3528ec7514e175b5cd0eb77fcc4d4531a7142b9e45a1ce6deb`

See:

- <https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner>
- <https://huggingface.co/onnx-community/mms-300m-1130-forced-aligner-ONNX>

## Packaged alignment code

Normalization, punctuation, repeated-token merging, and span behavior in `text.rb` and `ctc.rb` retain modified behavior from Fairseq MMS at revision `728b947019fd186753197add48c39cbb24ea43e2` and `ctc-forced-aligner` at revision `11855d1de76af2b490dd2e8e2db2661805ae90a0`. These bundled portions are distributed under CC-BY-NC-4.0; see `LICENSE.ctc-forced-aligner`.

The float32 CTC dynamic program in `ctc.rb` is a Ruby port of the TorchAudio 2.11.0 `forced_align` CPU kernel's documented state transitions and tie-breaking behavior. TorchAudio is BSD-2-Clause licensed; see `LICENSE.torchaudio`.

Romanization uses static, pure-Ruby single-codepoint data generated from the Uroman 1.3.1.1 reference for every character reachable from the pinned Cohere tokenizers, together with the reachable Greek and Japanese multi-codepoint/context rules. It does not invoke Python or an external romanizer at runtime. Uroman is written by Ulf Hermjakob, USC Information Sciences Institute (2015-2020), and is distributed under the permissive terms in `LICENSE.uroman`.
