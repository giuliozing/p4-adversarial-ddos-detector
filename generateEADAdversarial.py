import tensorflow as tf
import hickle as hkl
import numpy as np
import larq as lq
import os
from keras import backend as K

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

print("Loading model...")
# Layer personalizzati
custom_objects = {'DoReFa': lq.quantizers.DoReFa, 'f1_metric': f1_metric}

# Caricamento modello
model_path = os.path.join('modelliAddestrati', 'DDoSDetectorFullPrecision.h5')
original_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Costruzione modello wrapper che accetta (N, 4) come input
inp = tf.keras.Input(shape=(4,), dtype=tf.float32, name="wrapper_input")
split_inputs = [tf.expand_dims(inp[:, i], axis=-1) for i in range(4)]
out = original_model(split_inputs)
wrapped_model = tf.keras.Model(inputs=inp, outputs=out, name="wrapped_model")

print("Loading attack data...")
# Caricamento dati
data = hkl.load('dataNotNormalized_pcap.hkl')
X_attack = data['xattack'].astype(np.float32)  # (N, 4)
y_attack = data['yattack'].astype(np.int32)  # (N,)

print(f"Attack samples loaded: {len(X_attack)}")
print(f"Feature ranges in original data:")
print(f"  ack: [{X_attack[:, 0].min()}, {X_attack[:, 0].max()}]")
print(f"  seq: [{X_attack[:, 1].min()}, {X_attack[:, 1].max()}]")
print(f"  len: [{X_attack[:, 2].min()}, {X_attack[:, 2].max()}]")
print(f"  flags: [{X_attack[:, 3].min()}, {X_attack[:, 3].max()}]")

# EAD Attack con magnitude perturbation = 1
print("\n========== Custom Gradient-Based Attack ==========")
perturbation_magnitude = 5
print(f"Perturbation magnitude (L_inf): {perturbation_magnitude}")

print("\nGenerating adversarial examples with custom FGSM-like attack...")
print("This approach ensures L_inf constraint of 5...")

# Custom attack function with L_inf=5 constraint
def generate_adversarial_linf(model, x_original, epsilon=5, iterations=50, alpha=0.01):
    """
    Generate adversarial examples with L_inf constraint
    Args:
        model: Keras model
        x_original: Original samples
        epsilon: L_inf perturbation bound
        iterations: Number of iterations
        alpha: Step size
    """
    x_adv = tf.Variable(x_original, dtype=tf.float32)
    
    for i in range(iterations):
        if i % 10 == 0:
            print(f"  Iteration {i}/{iterations}...")
        
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            predictions = model(x_adv, training=False)
            # Loss: maximize probability of class 0 (normal) to evade detection
            loss = -predictions[:, 0]  # Negative to maximize
        
        # Get gradients
        gradients = tape.gradient(loss, x_adv)
        
        # Sign of gradients (like FGSM)
        perturbation = alpha * tf.sign(gradients)
        
        # Apply perturbation
        x_adv.assign(x_adv + perturbation)
        
        # Project back to L_inf ball
        delta = x_adv - x_original
        delta = tf.clip_by_value(delta, -epsilon, epsilon)
        x_adv.assign(x_original + delta)
        
        # Clip to valid ranges
        x_adv_clipped = tf.stack([
            tf.clip_by_value(x_adv[:, 0], 0, 4294967295),  # ack
            tf.clip_by_value(x_adv[:, 1], 0, 4294967295),  # seq
            tf.clip_by_value(x_adv[:, 2], 0, 15),           # data_offset
            tf.clip_by_value(x_adv[:, 3], 0, 63)            # flags
        ], axis=1)
        x_adv.assign(x_adv_clipped)
    
    return x_adv.numpy()

# Process in batches to avoid memory issues
batch_size = 1024
n_samples = len(X_attack)
n_batches = (n_samples + batch_size - 1) // batch_size

x_test_adv = np.zeros_like(X_attack)

print(f"\nProcessing {n_samples} samples in {n_batches} batches...")
for batch_idx in range(n_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, n_samples)
    
    print(f"\nBatch {batch_idx + 1}/{n_batches} (samples {start_idx}-{end_idx})...")
    x_batch = X_attack[start_idx:end_idx]
    
    x_adv_batch = generate_adversarial_linf(
        wrapped_model, 
        x_batch, 
        epsilon=perturbation_magnitude, 
        iterations=50,
        alpha=0.2
    )
    
    x_test_adv[start_idx:end_idx] = x_adv_batch

print("\n========== Post-processing for consistency ==========")

# Feature constraints:
# ack: 32-bit (0 to 2^32-1), but we use 8 MSB -> stored as full 32-bit value
# seq: 32-bit (0 to 2^32-1), but we use 8 MSB -> stored as full 32-bit value
# data_offset: 4-bit (0 to 15)
# flags: 6-bit (0 to 63)

print("Step 1: Applying L_inf perturbation constraint...")
# Calculate perturbations
perturbations = x_test_adv - X_attack
# Clip to magnitude constraint
perturbations = np.clip(perturbations, -perturbation_magnitude, perturbation_magnitude)
# Apply constrained perturbations
x_test_adv = X_attack + perturbations

print("Step 2: Rounding to integers...")
x_test_adv = np.round(x_test_adv)

print("Step 3: Clipping to valid ranges...")
x_test_adv[:, 0] = np.clip(x_test_adv[:, 0], 0, 4294967295)  # ack (32-bit)
x_test_adv[:, 1] = np.clip(x_test_adv[:, 1], 0, 4294967295)  # seq (32-bit)
x_test_adv[:, 2] = np.clip(x_test_adv[:, 2], 0, 15)          # data_offset (4-bit)
x_test_adv[:, 3] = np.clip(x_test_adv[:, 3], 0, 63)          # flags (6-bit)

print("Step 4: Converting to integer type...")
x_test_adv = x_test_adv.astype(np.int64)

# Verify constraints
print("\n========== Verification ==========")
print(f"Feature ranges in adversarial data:")
print(f"  ack: [{x_test_adv[:, 0].min()}, {x_test_adv[:, 0].max()}]")
print(f"  seq: [{x_test_adv[:, 1].min()}, {x_test_adv[:, 1].max()}]")
print(f"  len: [{x_test_adv[:, 2].min()}, {x_test_adv[:, 2].max()}]")
print(f"  flags: [{x_test_adv[:, 3].min()}, {x_test_adv[:, 3].max()}]")

# Check all values are non-negative
assert np.all(x_test_adv >= 0), "ERROR: Found negative values!"
print("✓ All values are non-negative")

# Check integer type
assert x_test_adv.dtype == np.int64, "ERROR: Not integer type!"
print("✓ All values are integers")

# Verify actual perturbations
actual_perturbations = np.abs(x_test_adv.astype(np.float32) - X_attack.astype(np.float32))
max_pert_per_feature = np.max(actual_perturbations, axis=0)
max_pert_overall = np.max(actual_perturbations)

print(f"\nActual perturbations applied:")
print(f"  Max perturbation per feature: ack={max_pert_per_feature[0]}, seq={max_pert_per_feature[1]}, len={max_pert_per_feature[2]}, flags={max_pert_per_feature[3]}")
print(f"  Max perturbation overall: {max_pert_overall}")
print(f"  L_inf norm: {max_pert_overall}")

# Count modified samples
modified_mask = np.any(actual_perturbations > 0, axis=1)
n_modified = np.sum(modified_mask)
print(f"\nSamples modified: {n_modified}/{len(X_attack)} ({100*n_modified/len(X_attack):.2f}%)")

# Save adversarial examples
print("\n========== Saving Results ==========")
output_filename = f'DDoSDetectorFullPrecision_EAD_mag{perturbation_magnitude}.hkl'
hkl.dump(x_test_adv, output_filename)
print(f"Adversarial examples saved to: {output_filename}")

# Evaluate evasion rate
print("\n========== Evaluation ==========")
print("Evaluating on original data...")
pred_original = wrapped_model.predict(X_attack, batch_size=256, verbose=0)
pred_original_labels = np.argmax(pred_original, axis=1)
acc_original = np.mean(pred_original_labels == y_attack)
print(f"Original accuracy (should detect as attack): {acc_original * 100:.2f}%")

print("\nEvaluating on adversarial data...")
pred_adv = wrapped_model.predict(x_test_adv.astype(np.float32), batch_size=256, verbose=0)
pred_adv_labels = np.argmax(pred_adv, axis=1)
acc_adv = np.mean(pred_adv_labels == y_attack)
print(f"Adversarial accuracy (still detected as attack): {acc_adv * 100:.2f}%")

evasion_rate = (1 - acc_adv) * 100
print(f"\n{'='*50}")
print(f"EVASION RATE: {evasion_rate:.2f}%")
print(f"{'='*50}")

# Save detailed results
results = {
    'x_adversarial': x_test_adv,
    'y_attack': y_attack,
    'perturbation_magnitude': perturbation_magnitude,
    'max_perturbation': max_pert_overall,
    'n_modified': n_modified,
    'original_accuracy': acc_original,
    'adversarial_accuracy': acc_adv,
    'evasion_rate': evasion_rate
}

results_filename = f'DDoSDetectorFullPrecision_EAD_mag{perturbation_magnitude}_results.hkl'
hkl.dump(results, results_filename)
print(f"\nDetailed results saved to: {results_filename}")

print("\n✓ Generation completed successfully!")
