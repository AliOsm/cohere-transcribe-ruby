# Audio segmentation attribution

The energy tokenizer in `segmentation.rb` retains the 50 ms PCM16 log-RMS calculation, silence state transitions, partial-final-frame timing, and maximum-duration behavior of Auditok 0.4.2 while implementing that behavior directly in Ruby.

- Source: <https://github.com/amsehili/auditok>
- PyPI source archive SHA-256: `52985096cbd3c15d650e71cb252b385875c9031da40ca8584b99fcdd9e26eaa5`
- Copyright: 2015-2026 Mohamed El Amine SEHILI
- License: MIT; see `LICENSE.auditok`
