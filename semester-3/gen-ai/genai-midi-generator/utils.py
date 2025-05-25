# utils.py
import pretty_midi
import numpy as np
import os


def midi_to_piano_roll(file_path, fs=8, n_notes=128, length=32):
    midi_data = pretty_midi.PrettyMIDI(file_path)
    piano_roll = midi_data.get_piano_roll(fs=fs)  # shape: (128, time)
    piano_roll = (piano_roll > 0).astype(np.float32)  # binarize

    if piano_roll.shape[1] < length:
        pad = length - piano_roll.shape[1]
        piano_roll = np.pad(piano_roll, ((0, 0), (0, pad)), mode='constant')
    else:
        piano_roll = piano_roll[:, :length]

    return piano_roll


def load_dataset(folder, max_files=100):
    X = []
    for i, file in enumerate(os.listdir(folder)):
        if file.endswith(".mid") or file.endswith(".midi"):
            roll = midi_to_piano_roll(os.path.join(folder, file))
            X.append(roll.flatten())
        if i >= max_files:
            break
    return np.array(X)


def piano_roll_to_midi(piano_roll, output_path, fs=8):
    import numpy as np
    import pretty_midi

    if len(piano_roll.shape) != 2:
        print("Error: piano_roll should be 2D. Got shape:", piano_roll.shape)
        return  # Avoid crashing

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for pitch in range(piano_roll.shape[0]):
        is_note_on = False
        start = 0
        for t in range(piano_roll.shape[1]):
            if piano_roll[pitch, t] > 0 and not is_note_on:
                start = t / fs
                is_note_on = True
            elif piano_roll[pitch, t] == 0 and is_note_on:
                end = t / fs
                note = pretty_midi.Note(
                    velocity=100, pitch=pitch, start=start, end=end)
                instrument.notes.append(note)
                is_note_on = False

    midi.instruments.append(instrument)
    midi.write(output_path)
