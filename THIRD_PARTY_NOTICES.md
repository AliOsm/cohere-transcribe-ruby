# Third-party notices

This source gem contains components under several licenses. The Apache License 2.0 in [`LICENSE.txt`](LICENSE.txt) applies to original cohere-transcribe Ruby code, native adapter code, and documentation. It does not replace the licenses identified below.

## Fairseq MMS and ctc-forced-aligner

Normalization, punctuation, repeated-token merging, and span behavior in `lib/cohere/transcribe/alignment/text.rb` and `lib/cohere/transcribe/alignment/ctc.rb` retain modified behavior from Fairseq MMS and ctc-forced-aligner.

- Fairseq MMS revision: `728b947019fd186753197add48c39cbb24ea43e2`
- Fairseq MMS source: <https://github.com/facebookresearch/fairseq/tree/728b947019fd186753197add48c39cbb24ea43e2/examples/mms>
- ctc-forced-aligner revision: `11855d1de76af2b490dd2e8e2db2661805ae90a0`
- ctc-forced-aligner source: <https://github.com/MahmoudAshraf97/ctc-forced-aligner/tree/11855d1de76af2b490dd2e8e2db2661805ae90a0>
- License: Creative Commons Attribution-NonCommercial 4.0 International; see [`lib/cohere/transcribe/alignment/LICENSE.ctc-forced-aligner`](lib/cohere/transcribe/alignment/LICENSE.ctc-forced-aligner)

The Fairseq source identifies Facebook, Inc. and its affiliates as the copyright holder. The ctc-forced-aligner source identifies Mahmoud Ashraf as its author. The Ruby implementation removes unsupported language and split modes, unused metadata, confidence aggregation, model-loading helpers, and native-extension integration while retaining the Arabic/English behavior used by this gem.

## TorchAudio

The float32 CTC dynamic program in `lib/cohere/transcribe/alignment/ctc.rb` is a Ruby port of the TorchAudio 2.11.0 `forced_align` CPU kernel's state transitions and tie-breaking behavior.

- Source: <https://github.com/pytorch/audio/blob/v2.11.0/src/libtorchaudio/forced_align/cpu/compute.cpp>
- Copyright: 2017 Facebook Inc. (Soumith Chintala)
- License: BSD-2-Clause; see [`lib/cohere/transcribe/alignment/LICENSE.torchaudio`](lib/cohere/transcribe/alignment/LICENSE.torchaudio)

## Uroman

`lib/cohere/transcribe/alignment/uroman_data.rb` and the Uroman rules in `lib/cohere/transcribe/alignment/text.rb` are generated or ported from Uroman 1.3.1.1.

- Source: <https://github.com/isi-nlp/uroman>
- Copyright: 2015-2020 Ulf Hermjakob, USC Information Sciences Institute
- License: the Uroman permissive license; see [`lib/cohere/transcribe/alignment/LICENSE.uroman`](lib/cohere/transcribe/alignment/LICENSE.uroman)

This project uses the universal romanizer software "uroman" written by Ulf Hermjakob, USC Information Sciences Institute (2015-2020). See the packaged license for the requested publication acknowledgement and bibliography.

## Auditok

The energy tokenizer in `lib/cohere/transcribe/audio/segmentation.rb` retains the 50 ms PCM16 energy and state-transition behavior of Auditok 0.4.2 while implementing it directly in Ruby.

- Source: <https://github.com/amsehili/auditok>
- Copyright: 2015-2026 Mohamed El Amine SEHILI
- License: MIT; see [`lib/cohere/transcribe/audio/LICENSE.auditok`](lib/cohere/transcribe/audio/LICENSE.auditok)

## Silero VAD and faster-whisper

`lib/cohere/transcribe/vad/silero_vad_v6.onnx`, the Ruby timestamp state machine, and the sequence runner use Silero VAD 6.2.1 behavior and the exact ONNX export distributed by faster-whisper.

- Silero VAD revision: `7e30209a3e901f9842f81b225f3e93d8199902b1`
- Silero VAD source: <https://github.com/snakers4/silero-vad/tree/v6.2.1>
- faster-whisper revision: `ed9a06cd89a93e47838f564998a6c09b655d7f43`
- faster-whisper source: <https://github.com/SYSTRAN/faster-whisper/tree/ed9a06cd89a93e47838f564998a6c09b655d7f43>
- License: MIT; see [`lib/cohere/transcribe/vad/LICENSE.silero-vad`](lib/cohere/transcribe/vad/LICENSE.silero-vad), [`lib/cohere/transcribe/vad/LICENSE.faster-whisper`](lib/cohere/transcribe/vad/LICENSE.faster-whisper), and [`lib/cohere/transcribe/vad/ATTRIBUTION.md`](lib/cohere/transcribe/vad/ATTRIBUTION.md)

## CrispASR and GGML

The native runtime vendors a deliberately reduced CrispASR 0.8.9 snapshot and its nested GGML snapshot. The gem adds a Ruby-facing session adapter, audio adapter, bounded batching, cancellation, and packaging changes while retaining the upstream source licenses and author records.

- CrispASR revision: `a68dd64092c4c44b41799a15976f5c6d27af13d7`
- CrispASR source: <https://github.com/CrispStrobe/CrispASR/tree/a68dd64092c4c44b41799a15976f5c6d27af13d7>
- GGML revision: `0714117daca2471b00e09554c7eaa74a06b0b2c5`
- GGML source: <https://github.com/CrispStrobe/ggml/tree/0714117daca2471b00e09554c7eaa74a06b0b2c5>
- License: MIT; see [`vendor/crispasr/LICENSE`](vendor/crispasr/LICENSE), [`vendor/crispasr/ggml/LICENSE`](vendor/crispasr/ggml/LICENSE), and [`vendor/crispasr/UPSTREAM.md`](vendor/crispasr/UPSTREAM.md)

## Runtime downloads and separately installed software

Model weights downloaded at runtime are not included in the gem and retain their own terms. The default Cohere ASR model is licensed under Apache License 2.0. The default MMS alignment model is licensed under Creative Commons Attribution-NonCommercial 4.0 International. Custom models and adapters remain subject to the terms published by their owners.

Ruby gems installed as dependencies and system libraries loaded at runtime are not copied into this source gem and retain their respective licenses.
