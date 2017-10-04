import click
import glob
import mido
import numpy
import os

notes = 24


def chords_from_midi(midi_file):
    chords = []
    notes_on = []
    midi = mido.MidiFile(midi_file)
    for track in midi.tracks:
        if track.name == 'Chords':
            previous_type = None
            for message in track:
                if message.type == 'note_on':
                    note = message.note
                    if note not in notes_on:
                        notes_on.append(note)
                if message.type == 'note_off':
                    if previous_type == 'note_on':
                        chords.append(notes_on)
                    note = message.note
                    if note in notes_on:
                        notes_on = notes_on[:]
                        notes_on.remove(note)
                if message.type == 'end_of_track':
                    chords.append(notes_on)
                previous_type = message.type
    return chords


def encode_chords(chords):
    encoded = numpy.zeros((len(chords), notes))
    minimum = min(min(x) for x in chords)
    for i, chord in enumerate(chords):
        for note in chord:
            encoded[i, note - minimum] = 1
    return encoded


def load_chords(midi_dir):
    progressions = []
    midi_files = sorted(glob.glob(os.path.join(midi_dir, '*.mid')) + glob.glob(os.path.join(midi_dir, '*.midi')))
    for midi_file in midi_files:
        try:
            chords = chords_from_midi(midi_file)
            encoded = encode_chords(chords)
            progressions.append(encoded)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            print("Skipping", midi_file)
    return progressions


def make_dataset_chords(midi_dir, output_file):
    data = load_chords(midi_dir)
    numpy.savez_compressed(output_file, data)
