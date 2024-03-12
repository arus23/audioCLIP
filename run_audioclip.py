
import os
import sys
import glob

import librosa
import librosa.display
import json
import numpy as np

import torch
import torchvision as tv

import pyaudio
import wave

from model import AudioCLIP
from utils.transforms import ToTensor1D

def record_audio(format, channels, rate, chunk, record_seconds = 12):
    FILENAME = "audio_file"
    p = pyaudio.PyAudio()

    stream = p.open(format = format, 
                channels = int(channels), 
                rate = int(rate), 
                input = True,
                frames_per_buffer = int(chunk))
    
    counter = 1
    print("started recording audio...")
    try:
        while(True):
            
            frames = []
            for _ in range(int(rate / chunk * record_seconds)):

                # Read audio data from the stream
                raw_data = stream.read(chunk)
                frames.append(raw_data)

            data = np.frombuffer(b''.join(frames), dtype=np.int16)

            audio_filename = f"test_audio/{FILENAME}_{counter}.wav"
            wf = wave.open(audio_filename, "wb")
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paFloat32))
            wf.setframerate(rate)
            wf.writeframes(data)
            wf.close()
            counter +=1

    except KeyboardInterrupt:
        print('Terminating recording...')

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

    return 


def main():
    sys.path.append(os.path.abspath(f'{os.getcwd()}/..'))

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)

    # Load all configurations       
    print("\nLoading config...")
    config = json.load(open("./audioclip-esc50.json"))

    # model_class = config['Model']['class']
    model_args = config['Model']['args']
    dataset_args = config['Dataset']['args']
    labels = config['Labels']

    MODEL_FILENAME = model_args['pretrained']
    AUDIO_ROOT = dataset_args['root']
    RATE = dataset_args['sample_rate']
    CHANNELS = dataset_args['channels']
    RECORD_SECONDS = 12
    CHUNK = dataset_args['chunk']
    FORMAT = pyaudio.paInt16
    LENGTH = dataset_args['length']
    IMAGE_SIZE = 224
    IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
    IMAGE_STD = 0.26862954, 0.26130258, 0.27577711
    LABELS = labels['list']

    # Model Instantiation
    aclp = AudioCLIP(pretrained = MODEL_FILENAME)

    # Load transforms
    audio_transforms = ToTensor1D()

    # image_transforms = tv.transforms.Compose([
    #     tv.transforms.ToTensor(),
    #     tv.transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
    #     tv.transforms.CenterCrop(IMAGE_SIZE),
    #     tv.transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
    # ])

    # record audio
    record_audio(FORMAT, CHANNELS, RATE, CHUNK, RECORD_SECONDS)

    # audio loading
    print("Loading audio files...")
    paths_to_audio = glob.glob("D:/TUK/Thesis/Projects/AudioCLIP/AudioCLIP-master/test_audio/*.wav")

    audio = []
    for path_to_audio in paths_to_audio:
        # track = 1D numpy array of audio wave form, _ = samppling rate 
        track, _ = librosa.load(path_to_audio, sr=RATE, dtype=np.float32)

        # Ensure tensor is of fixed length
        if len(track) < LENGTH:
            track = torch.nn.functional.pad(track, (0, LENGTH - len(track)))
        elif len(track) > LENGTH:
            track = track[:LENGTH]

        # compute spectrograms using trained audio-head (fbsp-layer of ESResNeXt)
        # thus, the actual time-frequency representation will be visualized
        spec = aclp.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)))
        spec = np.ascontiguousarray(spec.numpy()).view(np.complex64)
        pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()

        audio.append((track, pow_spec))

    # # image loading
    # paths_to_images = glob.glob('./demo/images/*.jpg')

    # images = []
    # for path_to_image in paths_to_images:
    #     with open(path_to_image, 'rb') as jpg:
    #         image = simplejpeg.decode_jpeg(jpg.read())
    #         images.append(image)

    # AudioCLIP handles raw audio on input, so the input shape is [batch x channels x duration]
    audio = torch.stack([audio_transforms(track.reshape(1, -1)) for track, _ in audio])
    # standard channel-first shape [batch x channels x height x width]
    # images = torch.stack([image_transforms(image) for image in images])
    # textual input is processed internally, so no need to transform it beforehand
    text = [[label] for label in LABELS]

    ## Obtaining Embeddings
    # AudioCLIP's output: Tuple[Tuple[Features, Logits], Loss]
    # Features = Tuple[AudioFeatures, ImageFeatures, TextFeatures]
    # Logits = Tuple[AudioImageLogits, AudioTextLogits, ImageTextLogits]

    ((audio_features, _, _), _), _ = aclp(audio=audio)
    # ((_, image_features, _), _), _ = aclp(image=images)
    ((_, _, text_features), _), _ = aclp(text=text)

    # Normalization of Embeddings
    # The AudioCLIP's output is normalized using L2-norm

    audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
    # image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)
    text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)

    ## Obtaining Logit Scales
    #Outputs of the text-, image- and audio-heads are made consistent using dedicated scaling terms for each pair of modalities.
    #The scaling factors are clamped between 1.0 and 100.0.
    scale_audio_image = torch.clamp(aclp.logit_scale_ai.exp(), min=1.0, max=100.0)
    scale_audio_text = torch.clamp(aclp.logit_scale_at.exp(), min=1.0, max=100.0)
    scale_image_text = torch.clamp(aclp.logit_scale.exp(), min=1.0, max=100.0)

    ## Computing Similarities
    # Similarities between different representations of a same concept are computed using [scaled](#Obtaining-Logit-Scales) dot product (cosine similarity).
    # logits_audio_image = scale_audio_image * audio_features @ image_features.T
    logits_audio_text = scale_audio_text * audio_features @ text_features.T
    # logits_image_text = scale_image_text * image_features @ text_features.T


    print('\t\tFilename, Audio\t\t\tTextual Label (Confidence)', end='\n\n')

    # calculate model confidence
    confidence = logits_audio_text.softmax(dim=1)
    for audio_idx in range(len(paths_to_audio)):
        # acquire Top-3 most similar results
        conf_values, ids = confidence[audio_idx].topk(1)

        # format output strings
        query = f'{os.path.basename(paths_to_audio[audio_idx]):>30s} ->\t\t'
        results = ', '.join([f'{LABELS[i]:>15s} ({v:06.2%})' for v, i in zip(conf_values, ids)])

        print(query + results)


if __name__ == '__main__':
    main()
