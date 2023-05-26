import torch, torchaudio
import argparse
import os
from model import MultiDecoderDPRNN

os.makedirs("outputs", exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--wav_file",
    type=str,
    default="",
    help="Path to the wav file to run model inference on.",
)
args = parser.parse_args()

mixture, sample_rate = torchaudio.load(args.wav_file)

model = MultiDecoderDPRNN.from_pretrained("JunzheJosephZhu/MultiDecoderDPRNN").eval()
if torch.cuda.is_available():
    model.cuda()
    mixture = mixture.cuda()
sources_est = model.separate(mixture).cpu()
for i, source in enumerate(sources_est):
  torchaudio.save(f"outputs/{i}.wav", source[None], sample_rate)