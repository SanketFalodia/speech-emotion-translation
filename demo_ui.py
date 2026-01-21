"""
Gradio UI for Emotion-Preserving Speech Translation
Hackathon Demo Interface
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import soundfile as sf

# Import pipeline
from emotion_pipeline import EmotionPreservingPipeline
from config import *


class DemoUI:
    def __init__(self):
        self.pipeline = EmotionPreservingPipeline()
        
    def create_emotion_chart(self, emotion, intensity):
        """Create visual emotion indicator"""
        fig, ax = plt.subplots(figsize=(6, 2))
        
        emotions = EMOTION_LABELS
        colors = ['red', 'brown', 'purple', 'yellow', 'gray', 'blue', 'orange']
        
        # Find emotion index
        try:
            idx = emotions.index(emotion)
        except:
            idx = 4  # neutral
        
        # Create bar chart
        values = [0] * len(emotions)
        values[idx] = intensity
        
        bars = ax.barh(emotions, values, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Confidence')
        ax.set_title(f'Detected Emotion: {emotion.upper()} ({intensity:.1%})')
        
        plt.tight_layout()
        return fig
    
    def create_prosody_comparison(self, original_prosody, modified_prosody):
        """Compare original vs modified prosody"""
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        
        metrics = ['Pitch', 'Speed', 'Energy']
        original_vals = [
            1.0,  # baseline
            1.0,
            1.0
        ]
        modified_vals = [
            modified_prosody['pitch_factor'],
            modified_prosody['speed_factor'],
            modified_prosody['energy_factor']
        ]
        
        for i, (ax, metric) in enumerate(zip(axes, metrics)):
            x = ['Original', 'Modified']
            y = [original_vals[i], modified_vals[i]]
            
            bars = ax.bar(x, y, color=['lightblue', 'lightcoral'])
            ax.set_ylabel('Factor')
            ax.set_title(metric)
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylim(0.5, 1.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}x',
                       ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def create_latency_breakdown(self, latency_dict):
        """Visualize latency breakdown"""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        steps = []
        times = []
        
        for step, time_val in latency_dict.items():
            if step != 'total':
                steps.append(step.replace('_', ' ').title())
                times.append(time_val)
        
        colors = ['skyblue', 'lightgreen', 'coral', 'gold', 'plum']
        bars = ax.barh(steps, times, color=colors[:len(steps)])
        
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Processing Latency Breakdown')
        ax.axvline(x=CONSTRAINTS['max_latency'], color='red', 
                  linestyle='--', label=f'Target: {CONSTRAINTS["max_latency"]}s')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.3f}s',
                   ha='left', va='center', fontsize=9)
        
        ax.legend()
        plt.tight_layout()
        return fig
    
    def process_audio(self, audio_input, source_lang, target_lang):
        """
        Main processing function called by Gradio
        """
        if audio_input is None:
            return (
                "❌ No audio provided",
                "",
                "",
                None,
                None,
                None,
                None,
                None,
                "No data"
            )
        
        try:
            # Process
            results = self.pipeline.process(
                audio_file=audio_input,
                source_lang=source_lang,
                target_lang=target_lang
            )
            
            # Create visualizations
            emotion_chart = self.create_emotion_chart(
                results['emotion'],
                results['intensity']
            )
            
            prosody_chart = self.create_prosody_comparison(
                results['prosody_original'],
                results['prosody_modified']
            )
            
            latency_chart = self.create_latency_breakdown(
                results['latency']
            )
            
            # Format metrics
            metrics_text = f"""
### 📊 Processing Metrics

**Emotion Detection:**
- Detected: {results['emotion'].upper()}
- Intensity: {results['intensity']:.1%}

**Prosody Adjustments:**
- Pitch: {results['prosody_modified']['pitch_factor']:.2f}x
- Speed: {results['prosody_modified']['speed_factor']:.2f}x
- Energy: {results['prosody_modified']['energy_factor']:.2f}x

**Performance:**
- Total Latency: {results['latency']['total']:.3f}s
- Target: {CONSTRAINTS['max_latency']}s
- Status: {'✅ PASS' if results['latency']['total'] <= CONSTRAINTS['max_latency'] else '❌ FAIL'}
            """
            
            # Check if within latency constraint
            status = "✅ Within latency target!" if results['latency']['total'] <= CONSTRAINTS['max_latency'] else "⚠️ Exceeds latency target"
            
            return (
                results['original_text'],
                results['translated_text'],
                f"😊 {results['emotion'].upper()} ({results['intensity']:.1%})",
                emotion_chart,
                prosody_chart,
                latency_chart,
                audio_input,  # Original audio
                results['output_file'],  # Translated audio
                metrics_text
            )
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
            return (error_msg, "", "", None, None, None, None, None, "Error occurred")
    
    def create_interface(self):
        """Create Gradio interface"""
        
        with gr.Blocks(title=UI_CONFIG['title'], theme=gr.themes.Soft()) as demo:
            
            gr.Markdown(f"# {UI_CONFIG['title']}")
            gr.Markdown(UI_CONFIG['description'])
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎤 Input")
                    
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Record or Upload Audio"
                    )
                    
                    with gr.Row():
                        source_lang = gr.Dropdown(
                            choices=list(SUPPORTED_LANGUAGES.keys()),
                            value="en",
                            label="Source Language"
                        )
                        target_lang = gr.Dropdown(
                            choices=list(SUPPORTED_LANGUAGES.keys()),
                            value="hi",
                            label="Target Language"
                        )
                    
                    process_btn = gr.Button("🚀 Translate with Emotion", variant="primary")
                    
                    gr.Markdown("---")
                    
                    # Text outputs
                    original_text = gr.Textbox(
                        label="📝 Original Text",
                        lines=2
                    )
                    
                    translated_text = gr.Textbox(
                        label="🌐 Translated Text",
                        lines=2
                    )
                    
                    emotion_display = gr.Textbox(
                        label="😊 Detected Emotion",
                        lines=1
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Analysis")
                    
                    emotion_chart = gr.Plot(label="Emotion Detection")
                    prosody_chart = gr.Plot(label="Prosody Adjustments")
            
            with gr.Row():
                gr.Markdown("### 🔊 Audio Comparison")
            
            with gr.Row():
                original_audio = gr.Audio(
                    label="Original Audio",
                    type="filepath"
                )
                translated_audio = gr.Audio(
                    label="Translated Audio (Emotion-Preserved)",
                    type="filepath"
                )
            
            with gr.Row():
                latency_chart = gr.Plot(label="⏱️ Latency Breakdown")
            
            with gr.Row():
                metrics_display = gr.Markdown("### Metrics will appear here")
            
            # Connect button
            process_btn.click(
                fn=self.process_audio,
                inputs=[audio_input, source_lang, target_lang],
                outputs=[
                    original_text,
                    translated_text,
                    emotion_display,
                    emotion_chart,
                    prosody_chart,
                    latency_chart,
                    original_audio,
                    translated_audio,
                    metrics_display
                ]
            )
            
            # Examples
            gr.Markdown("---")
            gr.Markdown("### 💡 Tips")
            gr.Markdown("""
            - Speak clearly for 3-5 seconds
            - Try different emotions: happy, sad, angry, excited
            - Compare the original vs translated audio for emotion preservation
            - Check the latency to ensure <2.5s target is met
            """)
        
        return demo


# ============================================================================
# LAUNCH
# ============================================================================
def main():
    print("🚀 Starting Emotion-Preserving Speech Translation Demo...")
    
    # Create output directories
    Path(AUDIO_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    
    # Create and launch UI
    ui = DemoUI()
    demo = ui.create_interface()
    
    print("✅ Demo ready!")
    print(f"🌐 Opening browser at http://localhost:7860")
    
    demo.launch(
        share=True,  # Create public link for hackathon demo
        server_name="0.0.0.0",
        server_port=7860
    )


if __name__ == "__main__":
    main()