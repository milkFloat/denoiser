import wave
import struct
import sounddevice as sd

wavefile = wave.open('/Users/ashanahan/Documents/GitHub/denoiser/output.wav', 'r')
length = wavefile.getnframes()

stream_out = sd.OutputStream(
        device=device_out,
        samplerate=args.sample_rate,
        channels=2)#channels_out)

for i in range(0, length):
    wavedata = wavefile.readframes(1)
    data = struct.unpack("<h", wavedata)
    print(int(data[0]))