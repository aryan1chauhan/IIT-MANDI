# SpeechLens Data Licensing & Attribution

All audio recordings and text used for the real-speech baselines are strictly public domain or openly licensed to allow commercial and non-commercial derivative creation (flaw injection and distribution).

---

## Real Baseline Recordings (`data/real/baseline/`)

### 1. `speech1.wav`
* **Work**: *The Tell-Tale Heart*
* **Author**: Edgar Allan Poe (1809–1849)
* **Text Source**: [Project Gutenberg EBook #2148](https://www.gutenberg.org/ebooks/2148) (Public Domain in the USA and worldwide)
* **Audio Source**: [LibriVox Short Story Collection 001](https://archive.org/details/stories_001_librivox) (`telltale_heart_poe_dm.mp3`)
* **Reader**: David Clarke
* **License**: Public Domain (All LibriVox recordings are in the Public Domain; dedicated to the commons)
* **Extraction**: 45.00 seconds cut from 00:01:00 to 00:01:45, downmixed to mono 16 kHz PCM WAV.
* **Transcript**: Normalized text in `data/real/transcripts/speech1.txt` (78 words).
* **Alignment**: MMS_FA forced alignment stored in `data/real/alignments/speech1.json` and `data/real/baseline/speech1_labels.txt`.
