import time

from img2midi import image2midi
from music21.midi.realtime import StreamPlayer
from music21 import converter


image2midi('midi-images/pleasant-excited/Banjo-Kazooie_N64_Banjo-Kazooie_Boggys Igloo Happy_0_Piano_0.png')

orig = converter.parse('midi-files/pleasant-excited/Banjo-Kazooie_N64_Banjo-Kazooie_Boggys Igloo Happy_0.mid')

sp = StreamPlayer(orig)
sp.play()

time.sleep(10)

translated_orig = converter.parse('Banjo-Kazooie_N64_Banjo-Kazooie_Boggys Igloo Happy_0_Piano_0.mid')

sp = StreamPlayer(orig)
sp.play()
