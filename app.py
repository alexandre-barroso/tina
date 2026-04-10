from flask import Flask, request, render_template, jsonify
import sys
import os
import random
import time
import datetime
import json
import threading
import time
import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Add path to inference.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn.functional as F
from inference import infer, _infer_single_word, infer_with_pattern, inv_ph_vocab, preprocess_input, dataset, model, device, SAVED_MODEL_PATH, SRC_LEN, MAX_PHONEME_LEN
from neuron_activity import generate_neuron_activity, capture_word_activations
from latent_space import PhoneticLatentSpace

# Import debug_inference, but handle the case where load_cumulative_metrics might not exist
try:    
    from debug import debug_inference, load_cumulative_metrics
except ImportError:
    from debug import debug_inference
    
    # Create a placeholder if the function doesn't exist
    def load_cumulative_metrics():
        return {
            "phoneme_avg": 0.0,
            "syllable_avg": 0.0,
            "stress_avg": 0.0,
            "alignment_avg": 0.0,
            "total_samples": 0
        }

app = Flask(__name__)

# Add this after initializing Flask app
_cached_neuron_data = None
_neuron_data_lock = threading.Lock()

# Load words from wordlist_ptbr.txt at startup
WORDLIST_PATH = os.path.join(os.path.dirname(__file__), "wordlist_ptbr.txt")
with open(WORDLIST_PATH, "r", encoding="utf-8") as f:
    WORDS = [line.strip() for line in f if line.strip()]
    
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = BASE_DIR
ERROR_LOG = os.path.join(BASE_DIR, "error_log.txt")
METRICS_PATH = os.path.join(BASE_DIR, "cumulative_metrics.json")

# Cache to hold last used time slot and inference results
_cached_slot = None
_cached_word = None
_cached_transcription = None
_cached_sample_words = None
_cached_metrics = None

def debug_inference_thread():
    """Background thread to periodically run debug inference and update error_log.txt"""
    print("[INFO] Starting periodic debug inference thread")
    
    while True:
        # Get current time - use datetime.datetime.now() instead of datetime.now()
        now = datetime.datetime.now()
        
        # Calculate seconds until next 10-minute mark
        minutes_to_wait = 10 - (now.minute % 10)
        seconds_to_wait = minutes_to_wait * 60 - now.second
        
        # Sleep until the next 10-minute mark
        time.sleep(seconds_to_wait)
        
        # Run the debug inference
        try:
            print(f"[INFO] Running scheduled debug inference at {datetime.datetime.now()}")
            debug_inference(n_samples=50, log_path=ERROR_LOG)
            print(f"[INFO] Debug inference completed at {datetime.datetime.now()}")
        except Exception as e:
            print(f"[ERROR] Failed to run debug inference: {e}")
            
def get_default_metrics():
    """Return default metrics structure if actual metrics can't be loaded"""
    return {
        "phoneme_avg": 0.0,
        "syllable_avg": 0.0,
        "stress_avg": 0.0,
        "alignment_avg": 0.0,
        "total_samples": 0
    }

def get_cached_inference():
    global _cached_slot, _cached_word, _cached_transcription, _cached_sample_words, _cached_metrics, _cached_variant_count
    now = datetime.datetime.utcnow()
    current_slot = now.strftime('%Y-%m-%d-%H') + f"-{now.minute // 10}"
    
    if _cached_slot != current_slot:
        _cached_slot = current_slot
        _cached_word = random.choice(WORDS)
        
        try:
            _cached_transcription, _cached_variant_count = infer(_cached_word)
        except Exception as e:
            print(f"[ERROR] Error during inference: {e}")
            _cached_transcription = "/<error>/"
            _cached_variant_count = 1
        
        # Cache sample words for showcase section
        _cached_sample_words = []
        for _ in range(15):
            try:
                word = random.choice(WORDS)
                transcription, variant_count = infer(word)
                _cached_sample_words.append({
                    "word": word, 
                    "transcription": transcription,
                    "variant_count": variant_count
                })
            except Exception as e:
                print(f"[ERROR] Error generating sample: {e}")
                # Add a placeholder if inference fails
                _cached_sample_words.append({
                    "word": "exemplo", 
                    "transcription": "/<error>/",
                    "variant_count": 1
                })
    
    # Ensure metrics is never None
    if _cached_metrics is None:
        _cached_metrics = get_default_metrics()
        
    return _cached_word, _cached_transcription, _cached_sample_words, _cached_metrics, _cached_variant_count

def generate_neuron_data_thread():
    """Background thread to periodically generate neuron activity data"""
    global _cached_neuron_data
    
    while True:
        try:
            output_file = generate_neuron_activity(num_words=3, num_neurons_per_layer=40)
            with open(output_file, 'r') as f:
                with _neuron_data_lock:
                    _cached_neuron_data = json.load(f)
            print(f"[INFO] Generated new neuron activity visualization")
        except Exception as e:
            print(f"[ERROR] Failed to generate neuron activity: {e}")
        
        # Generate new data every 30 seconds
        time.sleep(30)

# Start the background thread after app initialization
neuron_thread = threading.Thread(target=generate_neuron_data_thread, daemon=True)
neuron_thread.start()

debug_thread = threading.Thread(target=debug_inference_thread, daemon=True)
debug_thread.start()

@app.route("/")
def index():
    word, transcription, sample_words, metrics, variant_count = get_cached_inference()
    seconds_left = 600 - (int(time.time()) % 600)
    
    # Get model checkpoint time
    checkpoint_path = os.path.join(PROJECT_DIR, "model", "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        try:
            checkpoint_time = datetime.datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
            time_since_update = (datetime.datetime.now() - checkpoint_time).total_seconds() / 3600  # hours
        except Exception:
            time_since_update = 0
    else:
        time_since_update = 0
    
    # Ensure sample_words is never None
    if sample_words is None:
        sample_words = []
        
    # Ensure metrics is never None
    if metrics is None:
        metrics = get_default_metrics()
    
    return render_template(
        "index.html",
        word=word,
        transcription=transcription,
        variant_count=variant_count, 
        sample_words=sample_words,
        metrics=metrics,
        seconds_left=seconds_left,
        hours_since_update=time_since_update
    )

@app.route("/log")
def log():
    log_path = os.path.join(BASE_DIR, "training.log")
    if not os.path.exists(log_path):
        return jsonify({"lines": ["(log ainda não disponível)"], "loss_data": []})
    
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-50:]
        
        # Extract loss values where possible for charting
        loss_data = []
        for i, line in enumerate(lines):
            if "Epoch" in line and "summary" in line:
                try:
                    epoch_num = int(line.split("Epoch")[1].split()[0])
                    
                    # Look ahead for loss lines (within next 5 lines)
                    for j in range(i+1, min(i+6, len(lines))):
                        if "Avg Syllable Loss" in lines[j]:
                            syl_loss = float(lines[j].split("=")[1].strip())
                        elif "Avg Stress Loss" in lines[j]:
                            str_loss = float(lines[j].split("=")[1].strip())
                        elif "Avg Phoneme Loss" in lines[j]:
                            phon_loss = float(lines[j].split("=")[1].strip())
                    
                    loss_data.append({
                        "epoch": epoch_num,
                        "syllable": syl_loss,
                        "stress": str_loss,
                        "phoneme": phon_loss
                    })
                except Exception:
                    pass  # Skip any parsing errors
                    
        return jsonify({
            "lines": [line.strip() for line in lines if line.strip()],
            "loss_data": loss_data
        })
    except Exception as e:
        return jsonify({"lines": [f"Error reading log: {str(e)}"], "loss_data": []})
    
# Modify the log2 route in app.py to handle the translated metric names
@app.route("/log2")
def log2():
    if not os.path.exists(ERROR_LOG):
        return jsonify({"lines": ["(log ainda não disponível)"], "metrics_data": {}})
    
    try:
        with open(ERROR_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Extract metrics for visualization
        metrics_data = {}
        for i, line in enumerate(lines):
            # Handle both English and Portuguese formats
            if "Taxa de erro em Fonemas" in line:
                try:
                    # Extract value between ":" and "("
                    metric_value = float(line.split(":")[1].split("(")[0].strip())
                    metrics_data["Phoneme Error"] = metric_value
                except Exception as e:
                    print(f"Error parsing phoneme metric: {e}")
                    
            elif "Taxa de erro em Sílabas" in line:
                try:
                    metric_value = float(line.split(":")[1].split("(")[0].strip())
                    metrics_data["Syllable Error"] = metric_value
                except Exception as e:
                    print(f"Error parsing syllable metric: {e}")
                    
            elif "Taxa de erro em Acentuação" in line:
                try:
                    metric_value = float(line.split(":")[1].split("(")[0].strip())
                    metrics_data["Stress Error"] = metric_value
                except Exception as e:
                    print(f"Error parsing stress metric: {e}")
                    
            elif "Taxa de erro em Alinhamento" in line:
                try:
                    metric_value = float(line.split(":")[1].split("(")[0].strip())
                    metrics_data["Alignment Error"] = metric_value
                except Exception as e:
                    print(f"Error parsing alignment metric: {e}")
                    
        return jsonify({
            "lines": [line.strip() for line in lines if line.strip()],
            "metrics_data": metrics_data
        })
    except Exception as e:
        return jsonify({"lines": [f"Error reading log: {str(e)}"], "metrics_data": {}})

@app.route("/update-word")
def update_word():
    """Endpoint to manually update the showcase word without waiting for the timer"""
    try:
        word = random.choice(WORDS)
        transcription, variant_count = infer(word)  # Now receiving both values
        return jsonify({
            "word": word,
            "transcription": transcription,
            "variant_count": variant_count  # Return the count to frontend
        })
    except Exception as e:
        return jsonify({
            "word": "erro",
            "transcription": f"/<erro: {str(e)[:30]}...>/",
            "variant_count": 1,  # Default for error case
            "error": str(e)
        })

@app.route("/neuron-activity")
def neuron_activity():
    global _cached_neuron_data
    with _neuron_data_lock:
        if _cached_neuron_data is None:
            return jsonify({"error": "No neuron data available yet"})
        return jsonify(_cached_neuron_data)
        
@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Accepts JSON payload {"text": "<any words>", "pattern_idx": <optional int>},
    returns {"transcription": infer(text)} or {"original": "...", "pattern_influenced": "..."} 
    or {"error": "..."}.
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    try:
        # Check if pattern_idx is provided
        if "pattern_idx" in data and data["pattern_idx"] is not None:
            pattern_idx = int(data["pattern_idx"])
            # Use infer_with_pattern to get both transcriptions
            result = infer_with_pattern(text, pattern_idx)
            return jsonify(result)
        else:
            # Use standard infer — returns (variants_string, variant_count)
            transcription, variant_count = infer(text)

            # Capture real neuron activations for the viz (best-effort).
            neuron_activity = None
            try:
                neuron_activity = capture_word_activations(text)
            except Exception as exc:
                print(f"[WARN] capture_word_activations failed: {exc}")

            return jsonify({
                "transcription":   transcription,
                "variant_count":   variant_count,
                "neuron_activity": neuron_activity,
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Define a path for saving latent space visualizations
LATENT_SPACE_DIR = os.path.join(PROJECT_DIR, "latent_space_viz")
os.makedirs(LATENT_SPACE_DIR, exist_ok=True)

# Initialize latent space module (ensure dimensions match your model)
latent_space_module = PhoneticLatentSpace(dim=256, num_patterns=32)

# Load saved latent space if available
LATENT_SPACE_PATH = os.path.join(BASE_DIR, "model", "latent_space_model.pth")
if os.path.exists(LATENT_SPACE_PATH):
    try:
        latent_space_module.load_state_dict(torch.load(LATENT_SPACE_PATH, map_location=torch.device("cpu")))
        print(f"Loaded latent space model from {LATENT_SPACE_PATH}")
    except Exception as e:
        print(f"Could not load latent space model: {e}")

@app.route("/latent-space")
def latent_space_viz():
    """Return data for latent space visualization"""
    # Get pattern vectors and usage counts
    pattern_vectors = latent_space_module.pattern_vectors.detach().cpu().numpy()
    usage_counts = latent_space_module.usage_count.detach().cpu().numpy()
    
    # Create 2D projections for visualization
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    
    pca_projection = pca.fit_transform(pattern_vectors)
    tsne_projection = tsne.fit_transform(pattern_vectors)
    
    # Get phoneme co-occurrence matrix
    cooccur = latent_space_module.phoneme_cooccurrence.detach().cpu().numpy()
    
    # Calculate phoneme probabilities (row-normalized co-occurrence)
    row_sums = cooccur.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    phoneme_probs = cooccur / row_sums
    
    # Only include common phonemes in the visualization
    phoneme_counts = cooccur.sum(axis=1)
    top_phonemes = np.argsort(phoneme_counts)[-30:][::-1]  # Top 30 phonemes
    
    # Build phoneme data for visualization
    phoneme_data = []
    for idx in top_phonemes:
        # Skip if idx is not in the phoneme vocabulary
        if idx >= 256:  # Assuming max 256 phonemes
            continue
            
        # Get frequent next phonemes for this one
        next_phonemes = []
        for next_idx in np.argsort(phoneme_probs[idx])[-5:][::-1]:  # Top 5 next phonemes
            if next_idx < 256 and phoneme_probs[idx, next_idx] > 0.05:
                next_phonemes.append({
                    "index": int(next_idx),
                    "probability": float(phoneme_probs[idx, next_idx])
                })
        
        phoneme_data.append({
            "index": int(idx),
            "count": int(phoneme_counts[idx]),
            "next_phonemes": next_phonemes
        })
    
    # Prepare data for frontend
    response_data = {
        "patterns": [
            {
                "id": i,
                "usage": float(usage_counts[i]),
                "pca": [float(pca_projection[i, 0]), float(pca_projection[i, 1])],
                "tsne": [float(tsne_projection[i, 0]), float(tsne_projection[i, 1])],
                "vector": [float(v) for v in pattern_vectors[i]]
            }
            for i in range(len(pattern_vectors))
        ],
        "phonemes": phoneme_data
    }
    
    return jsonify(response_data)

@app.route("/explore-latent", methods=["POST"])
def explore_latent():
    """Generate a transcription by manipulating the latent space"""
    data = request.get_json(silent=True)
    if not data or "word" not in data or "pattern_id" not in data:
        return jsonify({"error": "Missing required fields"}), 400

    word = data["word"].strip().lower()
    pattern_id = int(data["pattern_id"])

    if not word:
        return jsonify({"error": "Empty word"}), 400

    try:
        # Reload the seq2seq model on every request
        if os.path.exists(SAVED_MODEL_PATH):
            model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=device))
        else:
            return jsonify({"error": "Model not found"}), 500

        # Reload the latent-space model on every request
        if os.path.exists(LATENT_SPACE_PATH):
            latent_space_module.load_state_dict(
                torch.load(LATENT_SPACE_PATH, map_location=device)
            )
        else:
            print(f"[WARN] latent-space file missing at {LATENT_SPACE_PATH}")

        # Validate pattern_id
        if not (0 <= pattern_id < latent_space_module.num_patterns):
            return jsonify({"error": f"Invalid pattern ID: {pattern_id}"}), 400

        # Use the existing infer_with_pattern function to get both transcriptions
        # Explicitly pass the latent_space_module to fix scope issues
        result = infer_with_pattern(word, pattern_id, latent_space_module)

        return jsonify({
            "word": word,
            "original_transcription": result["original"],
            "pattern_transcription": result["pattern_influenced"],
            "pattern_id": pattern_id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

        
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000, ssl_context='adhoc')
    


