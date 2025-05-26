# utils.py
import pretty_midi
import numpy as np
import os


def estimate_key(midi_data):
    try:
        chroma = midi_data.get_chroma(fs=8)
        chroma_sum = np.sum(chroma, axis=1)
        major_profile = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        scores = []
        for i in range(12):
            shifted_major = np.roll(major_profile, i)
            shifted_minor = np.roll(minor_profile, i)
            scores.append((np.correlate(chroma_sum, shifted_major)[0], i))
            scores.append((np.correlate(chroma_sum, shifted_minor)[0], i + 12))
        best_score, key_idx = max(scores)
        return key_idx
    except:
        return 0  # Default to C major


def midi_to_piano_roll(file_path, fs=8, n_notes=128, length=32):
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)
        piano_roll = midi_data.get_piano_roll(fs=fs)
        piano_roll = (piano_roll > 0).astype(np.float32)
        if piano_roll.shape[1] < length:
            pad = length - piano_roll.shape[1]
            piano_roll = np.pad(
                piano_roll, ((0, 0), (0, pad)), mode='constant')
        else:
            piano_roll = piano_roll[:, :length]
        if piano_roll.shape != (n_notes, length):
            raise ValueError(
                f"Invalid piano roll shape: {piano_roll.shape}, expected ({n_notes}, {length})")
        return piano_roll
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def load_dataset(folder, max_files=1000):
    X, keys = [], []
    for i, file in enumerate(os.listdir(folder)):
        if file.endswith((".mid", ".midi")):
            file_path = os.path.join(folder, file)
            roll = midi_to_piano_roll(file_path)
            if roll is not None:
                X.append(roll)
                midi_data = pretty_midi.PrettyMIDI(file_path)
                key_idx = estimate_key(midi_data)
                keys.append(key_idx)
        if i >= max_files - 1:
            break
    if not X:
        raise ValueError("No valid MIDI files loaded")
    X = np.array(X)
    assert X.shape[1:] == (128, 32), f"Loaded data shape mismatch: {X.shape}"
    return X, np.array(keys)


def piano_roll_to_midi(piano_roll, output_path, fs=8, min_duration=2):
    if len(piano_roll.shape) == 1 and piano_roll.shape[0] == 128 * 32:
        print("Auto reshaping piano roll from (4096,) to (128, 32)")
        piano_roll = piano_roll.reshape(128, 32)
    if len(piano_roll.shape) != 2 or piano_roll.shape != (128, 32):
        print("Error: piano_roll should be shape (128, 32). Got:", piano_roll.shape)
        return
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
    time_step = 1.0 / fs
    for pitch in range(piano_roll.shape[0]):
        is_note_on = False
        start = 0
        for t in range(piano_roll.shape[1]):
            if piano_roll[pitch, t] > 0 and not is_note_on:
                start = t
                is_note_on = True
            elif (piano_roll[pitch, t] == 0 or t == piano_roll.shape[1] - 1) and is_note_on:
                end = t if piano_roll[pitch, t] == 0 else t + 1
                if end - start >= min_duration:
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=start * time_step,
                        end=end * time_step
                    )
                    instrument.notes.append(note)
                is_note_on = False
    midi.instruments.append(instrument)
    midi.write(output_path)
