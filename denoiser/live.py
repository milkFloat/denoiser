# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import argparse
import sys

import sounddevice as sd
import soundfile as sf
import torch

import queue
import threading

from .demucs import DemucsStreamer
from .pretrained import add_model_flags, get_model
from .utils import bold

def get_parser():
    parser = argparse.ArgumentParser(
        "denoiser.live",
        description="Performs live speech enhancement, reading audio from "
                    "the default mic (or interface specified by --in) and "
                    "writing the enhanced version to 'Soundflower (2ch)' "
                    "(or the interface specified by --out)."
        )
    parser.add_argument(
        "-i", "--in", dest="in_",
        help="name or index of input interface.")
    parser.add_argument(
        "-o", "--out", default="Soundflower (2ch)",
        help="name or index of output interface.")
    add_model_flags(parser)
    parser.add_argument(
        "--sample_rate", type=int, default=16_000,
        help="Sample rate")
    parser.add_argument(
        "--no_compressor", action="store_false", dest="compressor",
        help="Deactivate compressor on output, might lead to clipping.")
    parser.add_argument(
        "--device", default="cpu")
    parser.add_argument(
        "--dry", type=float, default=0.04,
        help="Dry/wet knob, between 0 and 1. 0=maximum noise removal "
             "but it might cause distortions. Default is 0.04")
    parser.add_argument(
        "-t", "--num_threads", type=int,
        help="Number of threads. If you have DDR3 RAM, setting -t 1 can "
             "improve performance.")
    parser.add_argument(
        "-f", "--num_frames", type=int, default=1,
        help="Number of frames to process at once. Larger values increase "
             "the overall lag, but will improve speed.")
    parser.add_argument(
        "-b", "--bg_filename",
        help="Background Audio File")
    parser.add_argument(
        "--bg_blocksize", type=int, default=2048,
        help='block size (default: %(default)s)')
    parser.add_argument(
        "--bg_buffersize", type=int, default=20,
        help='number of blocks used for buffering (default: %(default)s)')
    return parser


def parse_audio_device(device):
    if device is None:
        return device
    try:
        return int(device)
    except ValueError:
        return device


def query_devices(device, kind):
    try:
        caps = sd.query_devices(device, kind=kind)
    except ValueError:
        message = bold(f"Invalid {kind} audio interface {device}.\n")
        message += (
            "If you are on Mac OS X, try installing Soundflower "
            "(https://github.com/mattingalls/Soundflower).\n"
            "You can list available interfaces with `python3 -m sounddevice` on Linux and OS X, "
            "and `python.exe -m sounddevice` on Windows. You must have at least one loopback "
            "audio interface to use this.")
        print(message, file=sys.stderr)
        sys.exit(1)
    return caps

def bg_audio(device_out):
    global bg_q
    bg_q = queue.Queue(maxsize=args.bg_buffersize)
    bg_event = threading.Event()

    if args.bg_filename:
        with sf.SoundFile(args.bg_filename) as f:
            for _ in range(args.bg_buffersize):
                data = f.buffer_read(args.bg_blocksize, dtype='float32')
                if not data:
                    break
                bg_q.put_nowait(data)  # Pre-fill queue
            
            bg_stream = sd.RawOutputStream(
                samplerate=f.samplerate, blocksize=args.bg_blocksize,
                device=device_out, channels=f.channels, dtype='float32',
                callback=bg_callback, finished_callback=bg_event.set)

            with bg_stream:
                timeout = args.bg_blocksize * args.bg_buffersize / f.samplerate
                while data:
                    data = f.buffer_read(args.bg_blocksize, dtype='float32')
                    bg_q.put(data, timeout=timeout)
                bg_event.wait()  # Wait until playback is finished

def bg_callback(outdata, frames, time, status):
    # assert frames == args.bg_blocksize
    if status.output_underflow:
        print('Output underflow: increase blocksize?', file=sys.stderr)
        raise sd.CallbackAbort
    assert not status
    try:
        data = bg_q.get_nowait()
    except queue.Empty as e:
        print('Buffer is empty: increase buffersize?', file=sys.stderr)
        raise sd.CallbackAbort from e
    if len(data) < len(outdata):
        outdata[:len(data)] = data
        outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
        raise sd.CallbackStop
    else:
        outdata[:] = data 

def main():
    global args
    args = get_parser().parse_args()

    if args.num_threads:
        torch.set_num_threads(args.num_threads)

    device_out = parse_audio_device(args.out)
    caps = query_devices(device_out, "output")
    channels_out = min(caps['max_output_channels'], 2)

    model = get_model(args).to(args.device)
    model.eval()
    print("Model loaded.")
    streamer = DemucsStreamer(model, dry=args.dry, num_frames=args.num_frames)

    bg_audio_thread = threading.Thread(target=bg_audio, args=(device_out,), daemon=True)
    bg_audio_thread.start()

    device_in = parse_audio_device(args.in_)
    caps = query_devices(device_in, "input")
    channels_in = min(caps['max_input_channels'], 2)
    
    stream_in = sd.InputStream(
        device=device_in,
        samplerate=args.sample_rate,
        channels=channels_in)

    
    stream_out = sd.OutputStream(
        device=device_out,
        samplerate=args.sample_rate,
        channels=channels_out)

    stream_in.start()
    stream_out.start()

    first = True
    current_time = 0
    last_log_time = 0
    last_error_time = 0
    cooldown_time = 2
    log_delta = 10
    sr_ms = args.sample_rate / 1000
    stride_ms = streamer.stride / sr_ms
    print(f"Ready to process audio, total lag: {streamer.total_length / sr_ms:.1f}ms.")
    while True:
        try:
            if current_time > last_log_time + log_delta:
                last_log_time = current_time
                tpf = streamer.time_per_frame * 1000
                rtf = tpf / stride_ms
                print(f"time per frame: {tpf:.1f}ms, ", end='')
                print(f"RTF: {rtf:.1f}")
                streamer.reset_time_per_frame()

            length = streamer.total_length if first else streamer.stride
            first = False
            current_time += length / args.sample_rate
            frame, overflow = stream_in.read(length)
            frame = torch.from_numpy(frame).mean(dim=1).to(args.device)
            with torch.no_grad():
                out = streamer.feed(frame[None])[0]
            if not out.numel():
                continue
            if args.compressor:
                out = 0.99 * torch.tanh(out)
            out = out[:, None].repeat(1, channels_out)
            mx = out.abs().max().item()
            if mx > 1:
                print("Clipping!!")
            out.clamp_(-1, 1)
            out = out.cpu().numpy()
            underflow = stream_out.write(out)
            if overflow or underflow:
                if current_time >= last_error_time + cooldown_time:
                    last_error_time = current_time
                    tpf = 1000 * streamer.time_per_frame
                    print(f"Not processing audio fast enough, time per frame is {tpf:.1f}ms "
                          f"(should be less than {stride_ms:.1f}ms).")
        except KeyboardInterrupt:
            print("Stopping")
            break
    
    stream_out.stop()
    stream_in.stop()


if __name__ == "__main__":
    main()
