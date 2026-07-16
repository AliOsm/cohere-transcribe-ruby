# Installed-gem WER benchmark

This directory contains a standalone release benchmark. The inference workload is deliberately not executed by `bundle exec rake`, the normal Minitest suite, or CI: it requires an installed source gem, a multi-gigabyte Dense checkpoint cache, frozen external audio, and suitable native hardware. Repository-wide static checks may still inspect these source files.

`installed_gem_wer.py` invokes `installed_gem_worker.rb` in one isolated Ruby process under an explicit `GEM_HOME`. The Python wrapper imports the retained frozen-suite parser and scorer from the adjacent research workspace so Ruby and Python hypotheses are evaluated with the same Arabic normalizers and Levenshtein implementation.

The public-API recognizer comparison uses `vad: "none"`, `alignment: "none"`, and a `max_dur` larger than every selected clip. Every file is an independent public-API input: `batch_size` can batch segments from one file but does not form cross-file batches. Long files reach the native single-row processor-compatible 30-to-35-second chunk planner. Do not use the normal 30-second fixed-window default for this comparison, because it changes the ASR inputs.

The default `--lane public_api` measures the supported file-level API end to end. `--lane native_batch` is a separately labeled throughput/accuracy lane: it loads runtime components from the verified installed gem, length-orders files up to 35 seconds with one clip per native processor row in a true cross-file batch of at most 24 rows, and sends longer files through the native single-row processor-compatible chunk planner. Python length-B24 starts with 24 utterances and its processor can expand them into more than 24 feature rows, so the two B24 labels do not describe an identical batching schedule. The native-batch lane does not claim that the public API schedules files this way.

Example balanced-500 command:

```bash
python benchmark/installed_gem_wer.py \
  ../benchmark/manifests/wit_probe500.jsonl \
  --output-dir ../benchmark/results/installed_gem_ruby_012 \
  --run-name ruby_012_bf16_balanced500 \
  --gem-home /path/to/isolated/gem-home \
  --gem-artifact ./cohere-transcribe-0.1.2.gem \
  --native-library /path/to/libcrispasr.so \
  --audio-library /path/to/libcohere_audio.so \
  --reference python_bf16=../benchmark/results/final_probe500_maskcache_20260711/bf16_length_b24_projcache_maskcache_repstop.hypotheses.jsonl
```

Add `--lane native_batch --batch-size 24` to run the explicit B24 lane. Use distinct output directories and run names for public and native-batch results.

Every report records hashes for the gem artifact and both native libraries, verifies that the artifact matches `GEM_HOME/cache`, records the installed gem root and complete option set, and preserves a manifest/audio identity fingerprint, compact hypotheses, raw worker rows, logs, timing, WER/CER, and paired Python transcript disagreements. File-level decode failures remain explicit and are scored as empty hypotheses; the benchmark does not silently drop them. Use a new run name for each native library candidate so artifacts cannot be overwritten accidentally.

Focused tooling checks are separate from the gem suite:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest benchmark/test_installed_gem_wer.py
ruby benchmark/test_installed_gem_worker.rb
ruby -c benchmark/installed_gem_worker.rb
```
