# Speech Analysis Project (KWS) - Fall 2026

## Project Overview
This project aims to build a robust Keyword Spotting (KWS) system using the Google Speech Commands dataset. The goal is to accurately classify short audio commands even in the presence of background noise.

## Phase 1: Problem Understanding and EDA
- **Dataset:** `mini_speech_commands` (8 classes: down, go, left, no, right, stop, up, yes)
- **Data Audit:** 8000 intact audio files audited. Extracted 1750 unique Speaker IDs.
- **Data Quality:** Identified 80 low-energy files (silence/noise) for specialized handling.
- **EDA:** Extracted and visualized Waveforms and Mel-Spectrograms to analyze acoustic fingerprints.