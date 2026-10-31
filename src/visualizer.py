import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import display

class SpeechDataVisualizer:
    """کلاس مصورسازی و تحلیل اکتشافی داده‌ها (EDA)"""
    
    def __init__(self, df_audio: pd.DataFrame, extracted_dir: str, words: list):
        self.df_audio = df_audio
        self.extracted_dir = extracted_dir
        self.words = words

    def plot_single_sample(self, word_dir: str, first_word: str):
        """رسم شکل موج و طیف‌نگار برای اولین نمونه (معادل سلول ۴)"""
        sample_audio_path = os.path.join(word_dir, os.listdir(word_dir)[0])
        y, sr = librosa.load(sample_audio_path, sr=None)

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(y, sr=sr, color='blue')
        plt.title(f"Waveform of the word '{first_word}'")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        librosa.display.specshow(mel_spect_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Mel Spectrogram of the word '{first_word}'")
        plt.tight_layout()
        plt.show()

    def analyze_speaker_distribution(self):
        """تحلیل گویندگان و رسم هیستوگرام توزیع (معادل سلول ۶)"""
        print("📊 ۱. بررسی توزیع نمونه‌های گویندگان در کلاس‌های مختلف:")
        speaker_counts = self.df_audio['Speaker_ID'].value_counts()
        print(f"▫️ پُرکارترین گوینده، {speaker_counts.max()} نمونه صدا در این مجموعه دارد.")
        print(f"▫️ تعداد گویندگانی که فقط یک فایل دارند: {(speaker_counts == 1).sum()} نفر")

        top_speakers = speaker_counts.head(5).index
        speaker_class_matrix = pd.crosstab(self.df_audio['Speaker_ID'], self.df_audio['Word_Class'])
        display(speaker_class_matrix.loc[top_speakers])
        print("💡 نتیجه ممیزی گوینده: همان‌طور که در ماتریس بالا می‌بینید، یک گوینده کلمات مختلفی را ادا کرده است. این اثبات می‌کند که برای جلوگیری از نشت اطلاعات، باید تقسیم‌بندی داده‌ها را حتماً بر اساس Speaker_ID انجام دهیم نه به صورت تصادفی.")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(speaker_counts, bins=range(1, speaker_counts.max() + 2), color="steelblue", edgecolor="white")
        ax.axvline(speaker_counts.median(), color="darkorange", linestyle="--", label=f"Median = {speaker_counts.median():.0f}")
        ax.axvline(speaker_counts.mean(), color="firebrick", linestyle="--", label=f"Mean = {speaker_counts.mean():.2f}")
        ax.set_xlabel("Number of files per speaker")
        ax.set_ylabel("Number of speakers")
        ax.set_title("Distribution of files per speaker (Data Leakage Audit)")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_all_classes_eda(self, seed=42, n_samples=3, hop_length=160):
        """رسم طیف‌نگار برای ۳ نمونه تصادفی از هر ۸ کلاس با تکنیک سایه و خط‌چین (معادل سلول ۷)"""
        print("⏳ در حال پردازش و رسم امواج و طیف‌نگارها برای ۳ نمونه تصادفی از هر کلاس...")

        for word in self.words:
            fig, axes = plt.subplots(nrows=2, ncols=n_samples, figsize=(15, 6))
            fig.suptitle(f"Keyword: '{word.upper()}' - Waveforms & Log-Mel Spectrograms", fontsize=16, fontweight='bold')
            
            samples = self.df_audio[self.df_audio['Word_Class'] == word].sample(n=n_samples, random_state=seed)
            
            for j, (idx, row) in enumerate(samples.iterrows()):
                filepath = os.path.join(self.extracted_dir, word, row['Filename'])
                y, sr = librosa.load(filepath, sr=None)
                end_t = len(y) / sr
                
                ax_w = axes[0, j]
                librosa.display.waveshow(y, sr=sr, ax=ax_w, color='teal')
                ax_w.set_title(f"Speaker: {row['Speaker_ID']}\nDuration: {end_t:.2f}s", fontsize=11)
                ax_w.set_xlim(0, 1.0)
                ax_w.set_ylim(-0.8, 0.8) 
                
                ax_s = axes[1, j]
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=hop_length, n_mels=64)
                mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)
                librosa.display.specshow(mel_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax_s)
                ax_s.set_xlim(0, 1.0)
                
                if end_t < 1.0:
                    for ax in (ax_w, ax_s):
                        ax.axvspan(end_t, 1.0, color='lightgray', alpha=0.5)
                        ax.axvline(end_t, color='red', linestyle='--', linewidth=1.5)
                        
                if j == 0:
                    ax_w.set_ylabel("Amplitude")
                    ax_s.set_ylabel("Hz")
                else:
                    ax_w.set_ylabel("")
                    ax_s.set_ylabel("")
                    ax_s.set_yticks([])
                ax_w.set_xlabel("")
                ax_w.set_xticks([])
                
            plt.tight_layout()
            plt.show()
        print("✅ رسم نمودارهای مقایسه‌ای پیشرفته با موفقیت انجام شد!")

    def plot_anomalies(self):
        """رسم عیب‌یابی برای فایل‌های سکوت و کلیپ شده (معادل منطق سلول ۶ و ۸)"""
        print("🔍 در حال رسم نمودارهای عیب‌یابی برای فایل‌های ناهنجار (سکوت و اعوجاج)...")

        quiet_samples = self.df_audio.nsmallest(4, "RMS")
        clipped_samples = self.df_audio.nlargest(4, "Clip_Frac")
        
        print("⚠️ شناسایی ضعیف‌ترین فایل‌ها:")
        display(quiet_samples[['Word_Class', 'Filename', 'RMS', 'Clip_Frac']])
        print("\n⚠️ شناسایی گوش‌خراش‌ترین فایل‌ها:")
        display(clipped_samples[['Word_Class', 'Filename', 'RMS', 'Clip_Frac']])

        fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18, 14), gridspec_kw={"height_ratios": [1, 1.5, 1, 1.5]})
        fig.suptitle("Visual Inspection of Anomalies: Quietest vs. Most Clipped Files", fontsize=18, fontweight='bold')

        categories = [("Quietest (Absolute Silence / Noise)", quiet_samples),
                      ("Most Clipped (High Distortion)", clipped_samples)]

        for k, (name, subset) in enumerate(categories):
            for j, (idx, row) in enumerate(subset.iterrows()):
                filepath = os.path.join(self.extracted_dir, row['Word_Class'], row['Filename'])
                y, sr = librosa.load(filepath, sr=None)
                
                ax_w = axes[2 * k, j]
                ax_s = axes[2 * k + 1, j]
                
                librosa.display.waveshow(y, sr=sr, ax=ax_w, color='blue')
                title = f"Class: '{row['Word_Class']}'\nRMS: {row['RMS']:.5f} | Clip: {row['Clip_Frac']:.3f}"
                ax_w.set_title(title, fontsize=10, fontweight='bold')
                
                if k == 0:
                    ax_w.set_ylim([-0.05, 0.05])
                else:
                    ax_w.set_ylim([-1.05, 1.05])

                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, n_mels=64)
                mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)
                librosa.display.specshow(mel_db, sr=sr, hop_length=160, x_axis='time', y_axis='mel', ax=ax_s, vmin=-80, vmax=0)
                
                if j == 0:
                    ax_w.set_ylabel(name + "\nAmplitude", fontsize=11, fontweight="bold")
                    ax_s.set_ylabel("Hz")
                else:
                    ax_w.set_ylabel("")
                    ax_w.set_yticks([])
                    ax_s.set_ylabel("")
                    ax_s.set_yticks([])
                ax_w.set_xlabel("")
                ax_w.set_xticks([])

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()
        print("✅ رسم نمودار عیب‌یابی با موفقیت انجام شد.")