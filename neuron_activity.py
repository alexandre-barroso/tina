import os
import json
import time
import torch
import numpy as np
import random
from neuralnet import LinguisticSeq2Seq, Encoder, SyllableDecoder, StressDecoder, PhonemeDecoder
from inference import preprocess_input, dataset, model, SAVED_MODEL_PATH

# Folder to save the activity data
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "webapp", "static")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_neuron_activity(num_words=5, num_neurons_per_layer=50):

    # Get some random words
    with open(os.path.join(os.path.dirname(__file__), "webapp", "wordlist_ptbr.txt"), "r", encoding="utf-8") as f:
        all_words = [line.strip() for line in f if line.strip()]
    words = random.sample(all_words, min(num_words, len(all_words)))
    
    # Load the latest model
    model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    
    # Initialize dictionary to store activations
    activity_data = []
    
    # Process each word
    with torch.no_grad():
        for word in words:
            try:
                print(f"Processing word: {word}")
                
                # Structure to store activations that matches the architecture
                activations = {
                    # Encoder components
                    "encoder": {
                        "embedding": [],
                        "conv": [],
                        "gru": [],
                        "transformer": []
                    },
                    # Decoder components
                    "decoder": {
                        "syllable": {
                            "embedding": [],
                            "attention": [],
                            "gru": [],
                            "output": []
                        },
                        "phoneme": {
                            "embedding": [],
                            "attention": [],
                            "gru": [],
                            "output": []
                        },
                        "stress": {
                            "attention": [],
                            "gru": [],
                            "transformer": [],
                            "output": []
                        }
                    }
                }
                
                # Register hooks to capture activations from the model
                hooks = []
                
                # Define a hook generator function to avoid closure issues
                def make_hook(component_name, layer_name):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            act = output[0]
                        else:
                            act = output
                            
                        if hasattr(act, 'detach'):
                            # Get the shape of the activation tensor
                            shape_info = list(act.shape)
                            
                            # For each layer, flatten the activation and sample neurons
                            flat_act = act.detach().cpu().numpy()
                            if flat_act.ndim > 1:
                                # For multi-dimensional tensors, average across all dimensions except the last
                                while flat_act.ndim > 2:
                                    flat_act = np.mean(flat_act, axis=1)
                                    
                                # If still 2D, get the first dimension (batch) or average if needed
                                if flat_act.ndim == 2:
                                    flat_act = flat_act[0]  # Take the first batch item
                            
                            # Now flatten completely
                            flat_act = flat_act.flatten()
                            
                            # If there are too many neurons, sample a subset
                            if len(flat_act) > num_neurons_per_layer:
                                indices = np.random.choice(len(flat_act), num_neurons_per_layer, replace=False)
                                sampled_act = flat_act[indices]
                            else:
                                sampled_act = flat_act
                            
                            # Normalize to [0, 1] for visualization
                            min_val = np.min(sampled_act)
                            max_val = np.max(sampled_act)
                            if max_val > min_val:
                                normalized_act = (sampled_act - min_val) / (max_val - min_val)
                            else:
                                normalized_act = np.zeros_like(sampled_act)
                            
                            # Store the normalized activations by component and layer
                            parts = component_name.split('.')
                            if len(parts) == 1:  # Encoder
                                activations["encoder"][parts[0]].append(normalized_act.tolist())
                            elif len(parts) == 2:  # Decoder
                                activations["decoder"][parts[0]][parts[1]].append(normalized_act.tolist())
                    
                    return hook_fn
                
                # Register hooks for encoder components
                hooks.append(model.encoder.embedding.register_forward_hook(
                    make_hook("embedding", "Encoder Embedding")))
                hooks.append(model.encoder.conv.register_forward_hook(
                    make_hook("conv", "Encoder Convolution")))
                hooks.append(model.encoder.rnn.register_forward_hook(
                    make_hook("gru", "Encoder GRU")))
                hooks.append(model.encoder.transformer_layer.register_forward_hook(
                    make_hook("transformer", "Encoder Transformer")))
                
                # Register hooks for syllable decoder components
                hooks.append(model.syllable_decoder.embedding.register_forward_hook(
                    make_hook("syllable.embedding", "Syllable Decoder Embedding")))
                hooks.append(model.syllable_decoder.attn.register_forward_hook(
                    make_hook("syllable.attention", "Syllable Decoder Attention")))
                hooks.append(model.syllable_decoder.rnn.register_forward_hook(
                    make_hook("syllable.gru", "Syllable Decoder GRU")))
                hooks.append(model.syllable_decoder.fc_token.register_forward_hook(
                    make_hook("syllable.output", "Syllable Decoder Output")))
                
                # Register hooks for phoneme decoder components
                hooks.append(model.phoneme_decoder.embedding.register_forward_hook(
                    make_hook("phoneme.embedding", "Phoneme Decoder Embedding")))
                hooks.append(model.phoneme_decoder.attn.register_forward_hook(
                    make_hook("phoneme.attention", "Phoneme Decoder Attention")))
                hooks.append(model.phoneme_decoder.rnn.register_forward_hook(
                    make_hook("phoneme.gru", "Phoneme Decoder GRU")))
                hooks.append(model.phoneme_decoder.fc_out.register_forward_hook(
                    make_hook("phoneme.output", "Phoneme Decoder Output")))
                
                # Register hooks for stress decoder components
                hooks.append(model.stress_decoder.encoder_attention.register_forward_hook(
                    make_hook("stress.attention", "Stress Decoder Attention")))
                hooks.append(model.stress_decoder.rnn.register_forward_hook(
                    make_hook("stress.gru", "Stress Decoder GRU")))
                hooks.append(model.stress_decoder.transformer.register_forward_hook(
                    make_hook("stress.transformer", "Stress Decoder Transformer")))
                hooks.append(model.stress_decoder.stress_classifier.register_forward_hook(
                    make_hook("stress.output", "Stress Decoder Output")))
                
                # Prepare input
                src = preprocess_input(word.lower(), dataset.src_vocab, 30)
                
                # Create character encodings for visualization
                input_chars = list(word.lower())
                char_encodings = []
                
                for i, char in enumerate(input_chars):
                    encoding_values = [random.uniform(0.1, 0.9) for _ in range(5)]
                    char_encodings.append({
                        "char": char,
                        "position": i,
                        "encoded": encoding_values
                    })
                
                # Process word through the model
                try:
                    model_output = model(
                        src=src,
                        teacher_syllable_seq=None,
                        teacher_phoneme_seqs=None,
                        max_phoneme_len=12,
                        teacher_forcing_ratio=0.0
                    )
                    
                    # Extract predictions
                    stress_logits = model_output["stress_logits"][0]
                    syllable_logits = model_output["syllable_logits"][0]
                    phoneme_outputs = model_output["phoneme_outputs"][0]
                    
                    # Convert predictions to simple string representations for visualization
                    stress_prediction = "".join([str(int(torch.argmax(s).item())) for s in stress_logits])
                    syllable_prediction = "".join([str(int(torch.argmax(s).item() % 10)) for s in syllable_logits])
                    
                    # Create phoneme prediction string
                    phoneme_prediction = ""
                    for ph_seq in phoneme_outputs:
                        ph_seq_str = "".join([str(int(torch.argmax(p).item() % 10)) for p in ph_seq])
                        if phoneme_prediction:
                            phoneme_prediction += "."
                        phoneme_prediction += ph_seq_str
                    
                except Exception as e:
                    print(f"Error running model: {e}")
                    # Use synthetic predictions as fallback
                    stress_prediction = "".join(["1" if i == 0 else "0" for i in range(len(word))])
                    syllable_prediction = "".join(["1" if i % 2 == 0 else "0" for i in range(len(word))])
                    phoneme_prediction = "".join([c if c in "aeiou" else get_phoneme_replacement(c) for c in word])
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
                
                # Format transcription
                final_transcription = format_transcription(syllable_prediction, phoneme_prediction, stress_prediction)
                
                # Create visualization steps
                syllable_steps = generate_proper_steps(syllable_prediction, min_steps=5)
                phoneme_steps = generate_proper_steps(phoneme_prediction, min_steps=5)
                stress_steps = generate_proper_steps(stress_prediction, min_steps=5)
                
                # Create network architecture information
                network_architecture = {
                    "nodes": [],
                    "connections": [],
                    "layers": [
                        # Encoder layers
                        {"id": "enc_emb", "name": "Encoder Embedding", "type": "encoder", "x": 0.1, "height": 0.8, "nodes": 20},
                        {"id": "enc_conv", "name": "Encoder Conv", "type": "encoder", "x": 0.15, "height": 0.7, "nodes": 20},
                        {"id": "enc_gru", "name": "Encoder GRU", "type": "encoder", "x": 0.2, "height": 0.8, "nodes": 30},
                        {"id": "enc_trans", "name": "Encoder Transformer", "type": "encoder", "x": 0.25, "height": 0.6, "nodes": 20},
                        
                        # Syllable decoder layers
                        {"id": "syl_emb", "name": "Syllable Embedding", "type": "syllable", "x": 0.4, "height": 0.7, "nodes": 15},
                        {"id": "syl_attn", "name": "Syllable Attention", "type": "syllable", "x": 0.45, "height": 0.6, "nodes": 12},
                        {"id": "syl_gru", "name": "Syllable GRU", "type": "syllable", "x": 0.5, "height": 0.7, "nodes": 20},
                        
                        # Phoneme decoder layers
                        {"id": "pho_emb", "name": "Phoneme Embedding", "type": "phoneme", "x": 0.65, "height": 0.7, "nodes": 15},
                        {"id": "pho_attn", "name": "Phoneme Attention", "type": "phoneme", "x": 0.7, "height": 0.6, "nodes": 12},
                        {"id": "pho_gru", "name": "Phoneme GRU", "type": "phoneme", "x": 0.75, "height": 0.7, "nodes": 20},
                        
                        # Stress decoder layers
                        {"id": "str_attn", "name": "Stress Attention", "type": "stress", "x": 0.85, "height": 0.6, "nodes": 12},
                        {"id": "str_gru", "name": "Stress GRU", "type": "stress", "x": 0.9, "height": 0.7, "nodes": 15},
                        {"id": "str_out", "name": "Stress Output", "type": "stress", "x": 0.95, "height": 0.5, "nodes": 10}
                    ]
                }
                
                # Define connections between layers
                connections = [
                    # Encoder connections
                    {"from": "enc_emb", "to": "enc_conv"},
                    {"from": "enc_conv", "to": "enc_gru"},
                    {"from": "enc_gru", "to": "enc_trans"},
                    
                    # Encoder to decoders
                    {"from": "enc_trans", "to": "syl_attn"},
                    {"from": "enc_trans", "to": "pho_attn"},
                    {"from": "enc_trans", "to": "str_attn"},
                    
                    # Syllable decoder internal
                    {"from": "syl_emb", "to": "syl_gru"},
                    {"from": "syl_attn", "to": "syl_gru"},
                    {"from": "syl_gru", "to": "pho_emb"},
                    
                    # Phoneme decoder internal
                    {"from": "pho_emb", "to": "pho_gru"},
                    {"from": "pho_attn", "to": "pho_gru"},
                    {"from": "pho_gru", "to": "str_attn"},
                    
                    # Stress decoder internal
                    {"from": "str_attn", "to": "str_gru"},
                    {"from": "str_gru", "to": "str_out"}
                ]
                
                # Add connections to the architecture
                network_architecture["connections"] = connections
                
                # Format and flatten the activations for the existing frontend
                formatted_activations = {
                    "encoder": [],
                    "syllable": [],
                    "phoneme": [],
                    "stress": []
                }
                
                # Flatten encoder activations
                for layer in ["embedding", "conv", "gru", "transformer"]:
                    if activations["encoder"][layer]:
                        formatted_activations["encoder"].extend(activations["encoder"][layer])
                
                # Flatten decoder activations
                for decoder_type in ["syllable", "phoneme", "stress"]:
                    for layer_name in activations["decoder"][decoder_type]:
                        if activations["decoder"][decoder_type][layer_name]:
                            formatted_activations[decoder_type].extend(
                                activations["decoder"][decoder_type][layer_name])
                
                # Create word data
                word_data = {
                    "word": word,
                    "timestamp": time.time(),
                    "activations": formatted_activations,
                    "network_architecture": network_architecture,
                    "transformation": {
                        "input_characters": input_chars,
                        "char_encodings": char_encodings,
                        "syllable_steps": syllable_steps,
                        "phoneme_steps": phoneme_steps,
                        "stress_steps": stress_steps
                    },
                    "prediction": {
                        "syllables": syllable_prediction,
                        "phonemes": phoneme_prediction,
                        "stress": stress_prediction,
                        "transcription": final_transcription
                    }
                }
                
                activity_data.append(word_data)
                
            except Exception as e:
                print(f"[ERROR] Failed to process word '{word}': {e}")
                continue
    
    # If we failed to process any words, generate dummy data
    if not activity_data:
        print("WARNING: No words were processed successfully. Generating dummy data.")
        activity_data = generate_dummy_data(all_words)
    
    # Save to file
    output_file = os.path.join(OUTPUT_DIR, "neuron_activity.json")
    with open(output_file, "w") as f:
        json.dump(activity_data, f, indent=2)
    
    print(f"Neuron activity data saved for {len(activity_data)} words.")
    return output_file

def generate_dummy_data(all_words):
    """Generate detailed dummy data that resembles the output format."""
    activity_data = []
    
    # Generate several dummy examples
    for dummy_word in ["exemplo", "palavra", "transcrição", "fonética"][:2]:
        if dummy_word not in all_words:
            dummy_word = all_words[0]
        
        # Create dummy phonetic components
        dummy_phonemes = convert_to_phonemes(dummy_word)
        dummy_syllables = create_syllable_pattern(dummy_word)
        dummy_stress = create_stress_pattern(dummy_word)
        dummy_transcription = format_transcription(dummy_syllables, dummy_phonemes, dummy_stress)
        
        # Create character encodings
        dummy_encodings = []
        for i, c in enumerate(dummy_word):
            encoding_values = [0.1 + 0.8 * random.random() for _ in range(5)]
            dummy_encodings.append({
                "char": c,
                "position": i,
                "encoded": encoding_values
            })
        
        # Generate proper step sequences
        syllable_steps = generate_proper_steps(dummy_syllables, min_steps=5)
        phoneme_steps = generate_proper_steps(dummy_phonemes, min_steps=5)
        stress_steps = generate_proper_steps(dummy_stress, min_steps=5)
        
        # Create dummy network architecture
        network_architecture = {
            "nodes": [],
            "connections": [],
            "layers": [
                # Encoder layers
                {"id": "enc_emb", "name": "Encoder Embedding", "type": "encoder", "x": 0.1, "height": 0.8, "nodes": 20},
                {"id": "enc_conv", "name": "Encoder Conv", "type": "encoder", "x": 0.15, "height": 0.7, "nodes": 20},
                {"id": "enc_gru", "name": "Encoder GRU", "type": "encoder", "x": 0.2, "height": 0.8, "nodes": 30},
                {"id": "enc_trans", "name": "Encoder Transformer", "type": "encoder", "x": 0.25, "height": 0.6, "nodes": 20},
                
                # Syllable decoder layers
                {"id": "syl_emb", "name": "Syllable Embedding", "type": "syllable", "x": 0.4, "height": 0.7, "nodes": 15},
                {"id": "syl_attn", "name": "Syllable Attention", "type": "syllable", "x": 0.45, "height": 0.6, "nodes": 12},
                {"id": "syl_gru", "name": "Syllable GRU", "type": "syllable", "x": 0.5, "height": 0.7, "nodes": 20},
                
                # Phoneme decoder layers
                {"id": "pho_emb", "name": "Phoneme Embedding", "type": "phoneme", "x": 0.65, "height": 0.7, "nodes": 15},
                {"id": "pho_attn", "name": "Phoneme Attention", "type": "phoneme", "x": 0.7, "height": 0.6, "nodes": 12},
                {"id": "pho_gru", "name": "Phoneme GRU", "type": "phoneme", "x": 0.75, "height": 0.7, "nodes": 20},
                
                # Stress decoder layers
                {"id": "str_attn", "name": "Stress Attention", "type": "stress", "x": 0.85, "height": 0.6, "nodes": 12},
                {"id": "str_gru", "name": "Stress GRU", "type": "stress", "x": 0.9, "height": 0.7, "nodes": 15},
                {"id": "str_out", "name": "Stress Output", "type": "stress", "x": 0.95, "height": 0.5, "nodes": 10}
            ]
        }
        
        # Define connections between layers
        connections = [
            # Encoder connections
            {"from": "enc_emb", "to": "enc_conv"},
            {"from": "enc_conv", "to": "enc_gru"},
            {"from": "enc_gru", "to": "enc_trans"},
            
            # Encoder to decoders
            {"from": "enc_trans", "to": "syl_attn"},
            {"from": "enc_trans", "to": "pho_attn"},
            {"from": "enc_trans", "to": "str_attn"},
            
            # Syllable decoder internal
            {"from": "syl_emb", "to": "syl_gru"},
            {"from": "syl_attn", "to": "syl_gru"},
            {"from": "syl_gru", "to": "pho_emb"},
            
            # Phoneme decoder internal
            {"from": "pho_emb", "to": "pho_gru"},
            {"from": "pho_attn", "to": "pho_gru"},
            {"from": "pho_gru", "to": "str_attn"},
            
            # Stress decoder internal
            {"from": "str_attn", "to": "str_gru"},
            {"from": "str_gru", "to": "str_out"}
        ]
        
        network_architecture["connections"] = connections
        
        # Create dummy activation data
        dummy_data = {
            "word": dummy_word,
            "timestamp": time.time(),
            "activations": {
                "encoder": [[random.random() for _ in range(40)] for _ in range(3)],
                "syllable": [[random.random() for _ in range(40)] for _ in range(2)],
                "phoneme": [[random.random() for _ in range(40)] for _ in range(2)],
                "stress": [[random.random() for _ in range(40)] for _ in range(2)]
            },
            "network_architecture": network_architecture,
            "transformation": {
                "input_characters": list(dummy_word),
                "char_encodings": dummy_encodings,
                "syllable_steps": syllable_steps,
                "phoneme_steps": phoneme_steps,
                "stress_steps": stress_steps
            },
            "prediction": {
                "syllables": dummy_syllables,
                "phonemes": dummy_phonemes,
                "stress": dummy_stress,
                "transcription": dummy_transcription
            }
        }
        
        activity_data.append(dummy_data)
    
    return activity_data

def convert_to_phonemes(word):

    result = []
    for char in word.lower():
        result.append(get_phoneme_replacement(char))
    return "".join(result)

def create_syllable_pattern(word):
    """Create a plausible syllable pattern for a word."""
    # In Portuguese, syllables typically follow consonant-vowel patterns
    pattern = []
    vowels = "aeiouáàâãéêíóôõú"
    in_syllable = False
    
    for char in word.lower():
        if char in vowels:
            if not in_syllable:
                pattern.append('1')  # Mark start of syllable
                in_syllable = True
            else:
                pattern.append('0')
        else:
            pattern.append('0')
            # End syllable after consonant if not at start
            if in_syllable and pattern:
                in_syllable = False
    
    # Ensure we have at least one syllable
    if '1' not in pattern and pattern:
        pattern[0] = '1'
    
    return "".join(pattern)

def create_stress_pattern(word):
    """Create a plausible stress pattern for a word."""
    # In Portuguese, stress is typically on the penultimate syllable
    vowels = "aeiouáàâãéêíóôõú"
    vowel_positions = [i for i, c in enumerate(word.lower()) if c in vowels]
    
    if not vowel_positions:
        return "1" + "0" * (len(word) - 1) if word else "0"
    
    # Default stress pattern with 0s
    pattern = ["0"] * len(word)
    
    # If accent marks are present, they indicate stress
    for i, c in enumerate(word):
        if c in "áàâãéêíóôõú":
            pattern[i] = "1"
            return "".join(pattern)
    
    # If no accent marks, place stress on penultimate syllable
    if len(vowel_positions) >= 2:
        pattern[vowel_positions[-2]] = "1"
    else:
        # If only one vowel, stress it
        pattern[vowel_positions[0]] = "1"
    
    return "".join(pattern)

def get_phoneme_replacement(char):
    """Get a plausible phonetic replacement for a character."""
    replacements = {
        'b': 'b', 'c': 'k', 'd': 'd', 'f': 'f', 'g': 'g', 'h': '', 'j': 'ʒ',
        'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'p': 'p', 'q': 'k', 'r': 'ɾ',
        's': 's', 't': 't', 'v': 'v', 'w': 'w', 'x': 'ʃ', 'y': 'j', 'z': 'z',
        'ç': 's', 'á': 'a', 'à': 'a', 'â': 'a', 'ã': 'ã', 'é': 'e', 'ê': 'e',
        'í': 'i', 'ó': 'o', 'ô': 'o', 'õ': 'õ', 'ú': 'u', 'ü': 'u'
    }
    return replacements.get(char.lower(), char)

def format_transcription(syllable_str, phoneme_str, stress_str):

    if not phoneme_str:
        phoneme_str = "???"
        
    try:
        transcription = "/" + phoneme_str + "/"
        
        # If syllables are available, add dots for syllable boundaries
        if syllable_str and '1' in syllable_str:
            # Find positions of '1's which mark syllable boundaries
            syllable_positions = [i for i, c in enumerate(syllable_str) if c == '1']
            
            if syllable_positions:
                # Map syllable positions to phoneme positions
                phoneme_positions = []
                for syl_pos in syllable_positions:
                    # Calculate proportional position in phoneme string
                    phoneme_pos = min(
                        len(phoneme_str) - 1, 
                        max(0, int(syl_pos * len(phoneme_str) / len(syllable_str)))
                    )
                    phoneme_positions.append(phoneme_pos)
                
                # Insert syllable markers
                chars = list(phoneme_str)
                modified_chars = []
                
                for i, char in enumerate(chars):
                    # Add syllable boundary at calculated positions
                    if i > 0 and i in phoneme_positions:
                        modified_chars.append('.')
                    modified_chars.append(char)
                
                transcription = "/" + "".join(modified_chars) + "/"
        
        # If stress information is available, mark primary stress with 'ˈ'
        if stress_str and '1' in stress_str:
            # Find the first stress position
            stress_position = stress_str.find('1')
            
            # Calculate where to put the stress marker in the transcription
            stress_phoneme_position = min(
                len(phoneme_str) - 1, 
                max(0, int(stress_position * len(phoneme_str) / len(stress_str)))
            )
            
            # Find the syllable boundary before this position
            dot_position = transcription.rfind('.', 0, stress_phoneme_position + 2)
            
            if dot_position >= 0:
                # Insert stress marker before the syllable
                transcription = transcription[:dot_position] + 'ˈ' + transcription[dot_position:]
            elif '.' in transcription:
                # If no dot before the stress position, put it at the first syllable
                first_dot = transcription.find('.')
                if first_dot > 1:  # Ensure we're not at the very beginning
                    transcription = transcription[:first_dot] + 'ˈ' + transcription[first_dot:]
        
        return transcription
    except Exception as e:
        print(f"Error formatting transcription: {e}")
        return "/" + phoneme_str + "/"

def generate_proper_steps(sequence, min_steps=5):

    if not sequence:
        return ["?"] * min_steps
    
    # Make sure sequence is substantial enough
    if len(sequence) < min_steps:
        # Stretch the sequence by repeating or padding
        stretched = sequence
        while len(stretched) < min_steps + 2:  # Add some buffer
            stretched += sequence[-1] if sequence else "?"
        sequence = stretched
    
    # Calculate how many steps to generate
    num_steps = max(min_steps, min(10, len(sequence)))
    
    # Calculate step size for even distribution
    step_size = max(1, len(sequence) // num_steps)
    
    # Generate evenly distributed steps
    steps = []
    for i in range(1, num_steps + 1):
        # Calculate position proportionally 
        pos = min(len(sequence), round(i * len(sequence) / num_steps))
        steps.append(sequence[:pos])
    
    # Ensure we have at least min_steps
    while len(steps) < min_steps:
        steps.append(sequence)  # Just repeat the full sequence
    
    return steps

if __name__ == "__main__":
    try:
        output_path = generate_neuron_activity()
        print(f"[INFO] Generated new neuron activity visualization")
    except Exception as e:
        print(f"[ERROR] Failed to generate neuron activity: {e}")
        # Create empty file as fallback
        output_file = os.path.join(OUTPUT_DIR, "neuron_activity.json")
        with open(output_file, "w") as f:
            # Generate even more detailed dummy data for visualization
            dummy_data = []
            for word in ["exemplo", "fonética"]:
                phonemes = convert_to_phonemes(word)
                syllables = create_syllable_pattern(word)
                stress = create_stress_pattern(word)
                transcription = format_transcription(syllables, phonemes, stress)
                
                # Create network architecture information
                network_architecture = {
                    "nodes": [],
                    "connections": [],
                    "layers": [
                        # Encoder layers
                        {"id": "enc_emb", "name": "Encoder Embedding", "type": "encoder", "x": 0.1, "height": 0.8, "nodes": 20},
                        {"id": "enc_conv", "name": "Encoder Conv", "type": "encoder", "x": 0.15, "height": 0.7, "nodes": 20},
                        {"id": "enc_gru", "name": "Encoder GRU", "type": "encoder", "x": 0.2, "height": 0.8, "nodes": 30},
                        {"id": "enc_trans", "name": "Encoder Transformer", "type": "encoder", "x": 0.25, "height": 0.6, "nodes": 20},
                        
                        # Syllable decoder layers
                        {"id": "syl_emb", "name": "Syllable Embedding", "type": "syllable", "x": 0.4, "height": 0.7, "nodes": 15},
                        {"id": "syl_attn", "name": "Syllable Attention", "type": "syllable", "x": 0.45, "height": 0.6, "nodes": 12},
                        {"id": "syl_gru", "name": "Syllable GRU", "type": "syllable", "x": 0.5, "height": 0.7, "nodes": 20},
                        
                        # Phoneme decoder layers
                        {"id": "pho_emb", "name": "Phoneme Embedding", "type": "phoneme", "x": 0.65, "height": 0.7, "nodes": 15},
                        {"id": "pho_attn", "name": "Phoneme Attention", "type": "phoneme", "x": 0.7, "height": 0.6, "nodes": 12},
                        {"id": "pho_gru", "name": "Phoneme GRU", "type": "phoneme", "x": 0.75, "height": 0.7, "nodes": 20},
                        
                        # Stress decoder layers
                        {"id": "str_attn", "name": "Stress Attention", "type": "stress", "x": 0.85, "height": 0.6, "nodes": 12},
                        {"id": "str_gru", "name": "Stress GRU", "type": "stress", "x": 0.9, "height": 0.7, "nodes": 15},
                        {"id": "str_out", "name": "Stress Output", "type": "stress", "x": 0.95, "height": 0.5, "nodes": 10}
                    ]
                }
                
                # Define connections between layers
                connections = [
                    # Encoder connections
                    {"from": "enc_emb", "to": "enc_conv"},
                    {"from": "enc_conv", "to": "enc_gru"},
                    {"from": "enc_gru", "to": "enc_trans"},
                    
                    # Encoder to decoders
                    {"from": "enc_trans", "to": "syl_attn"},
                    {"from": "enc_trans", "to": "pho_attn"},
                    {"from": "enc_trans", "to": "str_attn"},
                    
                    # Syllable decoder internal
                    {"from": "syl_emb", "to": "syl_gru"},
                    {"from": "syl_attn", "to": "syl_gru"},
                    {"from": "syl_gru", "to": "pho_emb"},
                    
                    # Phoneme decoder internal
                    {"from": "pho_emb", "to": "pho_gru"},
                    {"from": "pho_attn", "to": "pho_gru"},
                    {"from": "pho_gru", "to": "str_attn"},
                    
                    # Stress decoder internal
                    {"from": "str_attn", "to": "str_gru"},
                    {"from": "str_gru", "to": "str_out"}
                ]
                
                network_architecture["connections"] = connections
                
                # Add dummy data with our enhanced architecture representation
                dummy_data.append({
                    "word": word,
                    "timestamp": time.time(),
                    "activations": {
                        "encoder": [[random.random() for _ in range(50)] for _ in range(3)],
                        "syllable": [[random.random() for _ in range(50)] for _ in range(2)],
                        "phoneme": [[random.random() for _ in range(50)] for _ in range(2)],
                        "stress": [[random.random() for _ in range(50)] for _ in range(2)]
                    },
                    "network_architecture": network_architecture,
                    "transformation": {
                        "input_characters": list(word),
                        "char_encodings": [
                            {"char": c, "position": i, "encoded": [random.random() for _ in range(5)]} 
                            for i, c in enumerate(word)
                        ],
                        "syllable_steps": generate_proper_steps(syllables, min_steps=5),
                        "phoneme_steps": generate_proper_steps(phonemes, min_steps=5),
                        "stress_steps": generate_proper_steps(stress, min_steps=5)
                    },
                    "prediction": {
                        "syllables": syllables,
                        "phonemes": phonemes,
                        "stress": stress,
                        "transcription": transcription
                    }
                })
            
            json.dump(dummy_data, f, indent=2)
        print(f"Created fallback neuron_activity.json file")


# ─────────────────────────────────────────────────────────────────────────────
# Live activation capture for a user-supplied word
# Called by /transcribe so the viz can animate the actual forward pass.
# ─────────────────────────────────────────────────────────────────────────────
def capture_word_activations(word, num_neurons=40):

    import numpy as np
    from inference import preprocess_input, dataset, model, SAVED_MODEL_PATH

    # ── use first token only for the viz (one word at a time) ──────────────
    token = word.strip().split()[0].lower()

    # ── raw activation buckets, matching the four frontend groups ──────────
    raw = {
        "encoder": [],   # enc embedding → conv → GRU → transformer
        "syllable": [],  # syl embedding → attention → GRU → output
        "phoneme":  [],  # ph  embedding → attention → GRU → output
        "stress":   [],  # stress attention → GRU → transformer → output
    }

    hooks = []

    def _make_hook(group):
        def _hook(module, inp, output):
            act = output[0] if isinstance(output, tuple) else output
            if not hasattr(act, "detach"):
                return
            arr = act.detach().cpu().numpy()
            # collapse to 1-D: average all dims except last feature dim
            while arr.ndim > 2:
                arr = arr.mean(axis=1)
            if arr.ndim == 2:
                arr = arr[0]          # take batch item 0
            arr = arr.flatten()
            # subsample to num_neurons
            if len(arr) > num_neurons:
                idx = np.linspace(0, len(arr) - 1, num_neurons, dtype=int)
                arr = arr[idx]
            # normalise to [0, 1]
            lo, hi = arr.min(), arr.max()
            arr = (arr - lo) / (hi - lo) if hi > lo else np.full_like(arr, 0.5)
            raw[group].append(arr.tolist())
        return _hook

    try:
        # ── encoder ────────────────────────────────────────────────────────
        hooks.append(model.encoder.embedding.register_forward_hook(
            _make_hook("encoder")))
        hooks.append(model.encoder.conv.register_forward_hook(
            _make_hook("encoder")))
        hooks.append(model.encoder.rnn.register_forward_hook(
            _make_hook("encoder")))
        hooks.append(model.encoder.transformer_layer.register_forward_hook(
            _make_hook("encoder")))
        # ── syllable decoder ───────────────────────────────────────────────
        hooks.append(model.syllable_decoder.embedding.register_forward_hook(
            _make_hook("syllable")))
        hooks.append(model.syllable_decoder.attn.register_forward_hook(
            _make_hook("syllable")))
        hooks.append(model.syllable_decoder.rnn.register_forward_hook(
            _make_hook("syllable")))
        hooks.append(model.syllable_decoder.fc_token.register_forward_hook(
            _make_hook("syllable")))
        # ── phoneme decoder ────────────────────────────────────────────────
        hooks.append(model.phoneme_decoder.embedding.register_forward_hook(
            _make_hook("phoneme")))
        hooks.append(model.phoneme_decoder.attn.register_forward_hook(
            _make_hook("phoneme")))
        hooks.append(model.phoneme_decoder.rnn.register_forward_hook(
            _make_hook("phoneme")))
        hooks.append(model.phoneme_decoder.fc_out.register_forward_hook(
            _make_hook("phoneme")))
        # ── stress decoder ─────────────────────────────────────────────────
        hooks.append(model.stress_decoder.encoder_attention.register_forward_hook(
            _make_hook("stress")))
        hooks.append(model.stress_decoder.rnn.register_forward_hook(
            _make_hook("stress")))
        hooks.append(model.stress_decoder.transformer.register_forward_hook(
            _make_hook("stress")))
        hooks.append(model.stress_decoder.stress_classifier.register_forward_hook(
            _make_hook("stress")))

        # ── forward pass ───────────────────────────────────────────────────
        src = preprocess_input(token, dataset.src_vocab, 30)
        with torch.no_grad():
            model(
                src=src,
                teacher_syllable_seq=None,
                teacher_phoneme_seqs=None,
                max_phoneme_len=12,
                teacher_forcing_ratio=0.0,
            )

    finally:
        for h in hooks:
            h.remove()

    # ── fallback: if a group captured nothing, fill with neutral values ────
    dummy = [[0.5] * num_neurons]
    return {
        "word": token,
        "activations": {
            "encoder":  raw["encoder"]  or dummy,
            "syllable": raw["syllable"] or dummy,
            "phoneme":  raw["phoneme"]  or dummy,
            "stress":   raw["stress"]   or dummy,
        },
    }
