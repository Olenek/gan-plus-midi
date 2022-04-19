from music21.midi.realtime import StreamPlayer
from music21 import converter


def play_music(midi_path):
    orig = converter.parse(midi_path)

    sp = StreamPlayer(orig)
    sp.play()


play_music('gan/generated/2/9.mid')
