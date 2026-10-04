# Final Year Project Documentation

## Neural Network Based Morse-Code Audio Decoder

| Item | Details |
| --- | --- |
| Project title | Neural Network Based Morse-Code Audio Decoder |
| Submitted by | **[Student name]** |
| Register number | **[Register number]** |
| Programme / department | **[Programme and department]** |
| Institution | **[Institution name]** |
| Academic year | **[Academic year]** |
| Project guide | **[Guide name]** |

> Replace the bracketed fields above before submitting the report.

---

# 1. Introduction

Morse code is a communication method in which letters, digits, and selected punctuation marks are represented by short and long signal elements, conventionally called dots and dashes. Although it was developed for telegraphy, Morse code is still useful in amateur radio, emergency communication, accessibility tools, and low-bandwidth signalling. Converting a received Morse audio signal into readable text is traditionally performed by a skilled listener or by a rule-based system that measures tone and silence durations. Such systems can be sensitive to changes in transmission speed, tone frequency, background noise, and imperfect timing.

This project presents a **neural-network-based Morse-code audio decoder**. The system accepts a WAV audio file, transforms it into a sequence of spectral features, and predicts the corresponding text directly. A feed-forward feature extractor followed by a Long Short-Term Memory (LSTM) network learns temporal patterns in the signal. Connectionist Temporal Classification (CTC) is used during training so that the model can learn from an audio sequence and its text transcription without requiring a manually aligned dot/dash label for every time frame.

The project also provides two ways to use the trained model: a command-line decoder and a desktop graphical interface. The GUI allows a user to select a WAV file, choose an output text file, and save the decoded result without typing commands.

## 1.1 Problem statement

Reliable Morse decoding is difficult when the input differs from an ideal signal. In practice, the sender may use different words-per-minute (WPM) rates and tone frequencies, dot and dash durations may vary, recording levels may change, and noise may be present. A fixed threshold or timing rule must be tuned carefully for these conditions. The problem addressed by this project is to develop an audio-to-text decoder that learns useful signal and timing characteristics from varied synthetic Morse examples.

## 1.2 Objectives

- Generate realistic training examples of Morse audio with varied transmission conditions.
- Convert audio into short-time spectral features suitable for machine learning.
- Train a neural network to map feature sequences to text sequences.
- Use CTC loss to avoid frame-by-frame target alignment.
- Decode WAV files through both a command-line interface and a desktop GUI.
- Provide a reusable trained model for local inference.

## 1.3 Scope

The current system supports upper-case English letters, digits, spaces, and selected punctuation symbols (`.`, `,`, `?`, `=`, and `+`). The bundled decoder operates on WAV input, resamples it to 2 kHz, and writes plain-text output. The model is designed around single-channel Morse tone audio. Live microphone capture, language correction, and a full beam-search decoder are outside the current implementation scope.

## 1.4 Need for the study

The continued relevance of Morse communication comes from its simplicity and resilience. A Morse message can be transmitted through a narrow audio channel and can remain intelligible in circumstances in which richer voice communication is difficult. It is also a useful educational example because it connects communication theory, digital signal processing, machine learning, and user-interface design in one manageable application. A decoder that handles only ideal timing has limited practical value. Recordings made on different devices can vary considerably in loudness, carrier frequency, sampling rate, and noise. Human operators also do not key every dot or dash at exactly the same duration.

The central need is therefore not merely to translate a known dot-and-dash sequence into letters; a table lookup can already do that. The harder task is to infer the sequence from raw audio while accommodating variation. A learning-based method is well suited to this task because examples can expose a model to the range of conditions it should recognise. Instead of specifying a separate threshold for every possible recording, the model learns statistical relationships between spectral energy, silence, symbol duration, and output characters.

This project is also useful as a proof of concept for sequential audio recognition. The same broad design pattern—short-time features, a sequence model, and sequence-level loss—can be adapted to other low-bandwidth signal recognition problems. Examples include simple alarm detection, machine-state sounds, coded beacons, and assistive communication interfaces. The project deliberately keeps the domain narrow so that the complete path from data generation to inference can be inspected and reproduced.

## 1.5 Method overview

The work begins with a Morse-code generator. For each text label, the generator looks up each character in a Morse dictionary, constructs a tone-and-silence envelope, modulates it with a sinusoidal carrier, adds noise, and produces a waveform. Randomised parameters provide many distinct representations of the same underlying coding rules. The generated waveform is divided into 20 ms windows and converted to a spectrogram. Every column of the spectrogram represents frequency information for one short point in time.

The network accepts these feature vectors in temporal order. Dense layers transform each spectral vector into a learned feature representation, and the LSTM connects information across successive frames. The output has one probability distribution per frame. It includes every supported output character plus a blank class. CTC compares the entire output sequence with the shorter target text during learning. Consequently, the training data need only provide the final text; it does not need to mark the exact frame at which every character begins and ends.

At deployment time, WAV audio follows an equivalent preprocessing path. The system converts multi-channel sound to mono, resamples it to 2 kHz, normalises the level, computes the spectrogram, and passes it to the trained model. The decoder chooses the highest-probability label at each frame, removes repeated neighbouring labels and blank labels, and returns a readable string. The same decoding function is called from the command line and from the graphical interface, avoiding duplication of the core inference logic.

## 1.6 Significance of the project

The project demonstrates that classical coded communication can be approached with modern neural sequence recognition. It has educational significance because every design choice is visible in a small codebase: the alphabet definition, waveform synthesis, feature extraction, model architecture, loss function, decoding method, and interface are separated into understandable modules. It has practical significance because the saved checkpoint allows decoding without a cloud service or an internet connection. This is beneficial for privacy, portability, and use in settings where connectivity is unavailable.

For students, the system provides a concrete introduction to the difference between classification and sequence transcription. A single frame cannot normally identify a complete Morse character; the relevant evidence is distributed over many frames. CTC and an LSTM make it possible to model that distribution without manually segmenting every training waveform. The result is a project that is both technically meaningful and accessible for demonstration.

## 1.7 Organisation of the report

Chapter 1 introduces the problem, objectives, scope, and significance. Chapter 2 reviews conventional Morse decoding, recurrent neural networks, CTC, and spectrogram-based recognition. Chapter 3 analyses the existing and proposed systems and defines requirements and feasibility. Chapter 4 describes the system architecture, data flow, model, and interface design. Chapter 5 explains the implementation and execution workflow. Chapter 6 discusses available results, limitations, evaluation methodology, conclusions, and future work. Chapter 7 records acknowledgements. References are provided at the end for the academic concepts used in the work.

---

# 2. Literature Review

## 2.1 Conventional Morse decoding

Traditional automatic Morse decoders use signal processing to find the carrier tone and estimate the duration of tone-on and tone-off intervals. These durations are compared with the expected relationship between a dot, dash, intra-character gap, character gap, and word gap. Rule-based decoding can work very well for clear and consistently timed signals, but it normally requires thresholds or calibration. It can become fragile when a recording contains noise, amplitude changes, frequency shifts, or irregular human keying.

## 2.2 Recurrent neural networks for sequences

Morse audio is inherently sequential: nearby audio frames combine to represent a dot, dash, pause, character, or word boundary. Recurrent neural networks retain information across time, making them appropriate for speech, handwriting, and other sequence-recognition tasks. The LSTM architecture introduced by Hochreiter and Schmidhuber uses gated memory cells to reduce the vanishing-gradient problem of basic recurrent networks [1]. This makes it suitable for learning timing-dependent patterns in Morse signals.

## 2.3 Connectionist Temporal Classification

Graves et al. proposed CTC for sequence labelling where input and output lengths differ and explicit alignment is unavailable [2]. CTC includes a blank label and sums over valid alignments during learning. At inference, repeated predictions are collapsed and blanks are removed. This directly matches Morse transcription: a character may occupy many spectrogram frames, while the expected target is a short text string. CTC has also been a major component of end-to-end speech recognition systems [3].

## 2.4 Spectrogram-based audio recognition

Audio waveforms are commonly converted into time-frequency representations before recognition. A spectrogram separates an audio stream into short windows and represents the frequency content of each window. It retains timing information while making the signal structure easier for a learning model to process. In this project, non-overlapping 20 ms windows are used. This choice produces an incremental stream of features and supports future streaming use because the model does not require a convolution over the complete audio sequence.

## 2.5 Research gap and project contribution

The reviewed approaches suggest that hand-crafted timing rules are simple but condition-sensitive, while recurrent neural models and CTC can learn sequence alignments from examples. The contribution of this project is a compact, reproducible implementation that synthesizes diverse Morse transmissions, learns an audio-to-text mapping, and provides an end-user decoding interface. Rather than requiring a separate dot/dash segmentation stage, the system predicts text from spectral frames.

## 2.6 Review of signal-processing concepts

An audio recording is a sequence of amplitude samples. Examining amplitude alone can be misleading because noise and level changes can make a fixed threshold unreliable. Frequency-domain analysis offers an additional view: a deliberately transmitted Morse tone tends to concentrate energy around a carrier frequency, whereas silence and random noise have different patterns. The short-time Fourier transform, represented here by a spectrogram, calculates frequency content over consecutive short intervals. It captures both *what frequencies* are present and *when* they are present.

The selection of window size is a trade-off. A long window gives fine frequency resolution but blurs short timing changes; a very short window captures timing changes but gives less frequency detail. The 20 ms window used in this project is a reasonable starting point for a low-frequency Morse tone at a 2 kHz sample rate. It yields a stream of non-overlapping observations, reduces computation, and makes it possible to process audio incrementally. A future study could compare window duration, overlap, mel-frequency features, and learned front ends using a common test corpus.

In conventional systems, signal processing is often followed by thresholding and duration estimation. For example, energy near a selected tone may be classified as “key down” when it exceeds a threshold. Runs of key-down frames can then be measured and assigned to dot or dash categories. This is interpretable and inexpensive, but each stage relies on assumptions: the carrier must be found, the threshold must be appropriate, and the timing distribution must be sufficiently regular. The neural method retains the useful spectral representation but learns the later decision stages from examples.

## 2.7 Review of training-data generation

Supervised learning requires pairs of inputs and correct outputs. Obtaining a large, carefully transcribed collection of real Morse recordings can be time-consuming. Synthetic generation is therefore a valuable approach for an initial model. Because Morse coding rules are known, a program can generate a waveform and its exact text transcription at the same time. This eliminates label ambiguity and allows controlled variation of one factor at a time.

Data augmentation is widely used in audio and speech recognition to make models less dependent on recording conditions. In this project, the generator varies pitch, words per minute, amplitude, noise power, and symbol timing. Variation in timing is especially important: a model trained only on perfectly uniform dots and dashes may not generalise to hand-keyed transmissions. Synthetic data does have a limitation: real signals may include radio fading, interference, clipping, reverberation, drift, compression, and non-Gaussian noise not represented in the generator. For that reason, synthetic training should be regarded as an efficient foundation, not a replacement for field evaluation.

## 2.8 Comparison of candidate approaches

| Approach | Strengths | Limitations | Relevance to this project |
| --- | --- | --- | --- |
| Manual decoding | Flexible human judgement; no software required | Slow and dependent on operator skill | Establishes the original task being automated |
| Timing-rule decoder | Interpretable, small, and fast | Requires threshold and timing calibration | Useful baseline for later comparison |
| Convolutional audio classifier | Learns local spectral patterns | Needs a method to model long sequence alignment | Could be investigated as a future front end |
| LSTM with CTC | Models temporal context and unaligned labels | Requires training data and careful evaluation | Selected approach |
| Transformer sequence model | Powerful global context modelling | More data and computation may be needed | Possible future extension |

The selected LSTM-CTC approach is appropriate for the project scale. It avoids the complexity of a large transformer while preserving temporal memory and alignment-free training. The dense layers reduce and reshape the per-frame feature representation before recurrent processing. This design is also relatively easy to explain during a final-year project demonstration.

## 2.9 Critical summary

The literature supports three key decisions in the present system. First, short-time spectral features provide a meaningful representation for coded audio. Second, LSTMs are appropriate when the interpretation of a frame depends on earlier and later context. Third, CTC permits direct training from audio and text without hand-aligning symbols. These ideas do not guarantee accuracy by themselves; their effectiveness depends on training coverage, model capacity, decoding strategy, and evaluation quality. The report therefore treats the bundled examples as a functional demonstration and explicitly distinguishes them from a formal accuracy claim. A rigorous extension should establish a baseline rule-based decoder, reserve a test set that is never used in training, and report error rates with confidence intervals across defined acoustic conditions.

---

# 3. System Analysis

## 3.1 Existing system

In a typical manual or rule-based workflow, a user listens to the audio or performs the following operations:

1. Detect the Morse tone and separate it from silence.
2. Measure signal and gap lengths.
3. classify each signal as a dot or dash.
4. Group symbols into characters using timing rules.
5. Convert the Morse symbols into text using a lookup table.

This approach is understandable and lightweight, but it relies on stable timing thresholds. An incorrect decision at an early stage can affect later character grouping.

## 3.2 Proposed system

The proposed system uses a learned pipeline:

```text
WAV audio
    ↓
Mono conversion, resampling to 2 kHz, peak normalization
    ↓
20 ms spectrogram frames
    ↓
Four dense ReLU layers
    ↓
LSTM sequence layer
    ↓
Character probabilities, including CTC blank
    ↓
Greedy CTC collapse → decoded text / .txt output
```

The model is trained using generated audio whose pitch, speed, amplitude, noise level, and dot/dash timing are varied. This encourages the network to recognize signal structure rather than memorize one ideal tone.

## 3.3 Functional requirements

| ID | Requirement |
| --- | --- |
| FR1 | The system shall generate Morse audio and its corresponding text label for training. |
| FR2 | The system shall accept a WAV file for decoding. |
| FR3 | The system shall convert stereo input to mono and resample input to 2 kHz. |
| FR4 | The system shall generate a spectrogram using 20 ms windows. |
| FR5 | The system shall load a saved PyTorch model and produce a text prediction. |
| FR6 | The command-line interface shall print the prediction or save it to a specified text file. |
| FR7 | The GUI shall allow selection of input audio and output text paths. |
| FR8 | The system shall report invalid, empty, or silent audio input. |

## 3.4 Non-functional requirements

The solution should run locally, provide deterministic file-based output, use commonly available Python libraries, and remain usable without specialised radio hardware. The GUI should be simple enough for a user who does not use the command line. Model inference uses the CPU, which makes the packaged decoder portable; GPU acceleration is used by the training program.

## 3.5 Feasibility analysis

**Technical feasibility:** Python, PyTorch, SciPy, and Tkinter provide the required tools for audio processing, modelling, and interface construction. A trained checkpoint is included in `models/001750.pt`.

**Operational feasibility:** The workflow is direct: choose an audio file, decode it, and read the generated text file. No network service is required during normal decoding.

**Economic feasibility:** The project is based on open-source software and can run on a standard computer. Training benefits from a CUDA-capable GPU, but inference is configured for CPU execution.

## 3.6 Stakeholders and use cases

The primary user is a person who has a Morse audio recording and wants a text transcription. This may be a student, hobbyist, demonstrator, or researcher. The secondary user is the project maintainer, who trains or replaces the model checkpoint. A third stakeholder is the evaluator, who needs a clear way to demonstrate the system and inspect its output.

The main use case begins when a user supplies a WAV file. The user either invokes the command-line program or opens the desktop application. The system validates the file, processes its audio, loads the saved model, and produces text. In the command-line path, text is printed to standard output unless an output path is specified. In the GUI path, the user chooses a destination file and receives a success or failure message. The training use case is separate: the developer runs `main.py`, monitors loss in TensorBoard, and obtains numbered checkpoint files.

Exceptional use cases are also important. A user may choose a non-existent path, an empty audio file, or a silent recording. The decoder contains explicit checks for the latter two cases and the GUI checks that the audio file exists. A model file may be missing or incompatible; in that case the exception is surfaced to the caller, and the GUI displays an error dialog. Such behaviour is preferable to silently generating misleading text.

## 3.7 Input, process, and output analysis

| Stage | Input | Processing | Output |
| --- | --- | --- | --- |
| Training generation | Random parameters and text | Build Morse envelope, modulate tone, add noise | Waveform, spectrogram, target string |
| Feature preparation | Waveform | Calculate 20 ms spectrogram and arrange time-major features | Tensor sequence |
| Network training | Padded feature and label batches | Forward pass, CTC loss, back-propagation, Adam update | Updated model weights |
| File decoding | WAV file and checkpoint | Preprocess, infer, greedy CTC collapse | Decoded text |
| User interaction | Selected paths | Validate paths, call decoder, write text | Saved `.txt` file and status |

This analysis shows that the system has a clear separation between offline learning and online inference. Training is computationally intensive and may run for many epochs, while decoding is a short local operation. Keeping the two paths separate makes the delivered application smaller and easier for an end user to operate.

## 3.8 Risk analysis and mitigation

| Risk | Possible effect | Mitigation in current work or future work |
| --- | --- | --- |
| Training examples are too idealised | Reduced real-world generalisation | Add real recordings and richer noise simulation |
| Unsupported or malformed WAV file | Decode failure | Validate input and present clear errors |
| Silent audio | Division by zero or meaningless output | Reject zero-peak input explicitly |
| GPU unavailable during training | Training cannot start | Add automatic CPU fallback in a future revision |
| Greedy decoder makes local mistakes | Higher transcription error | Implement CTC beam search |
| Output path is unwritable | Result cannot be saved | Catch and display write errors in the GUI |
| User expects unsupported characters | Missing or incorrect output | Publish the supported alphabet clearly |

## 3.9 Acceptance criteria

The system is considered functionally acceptable when it can generate labelled samples, load the supplied model, accept a valid non-silent WAV file, and return a text prediction without manual dot/dash segmentation. The command-line interface must support output-file writing, and the GUI must enable file selection and show errors rather than failing silently. For an academic evaluation, the implementation should additionally be demonstrated using the supplied example audio and at least several separately prepared recordings. A numerical performance target should only be introduced after a labelled held-out evaluation corpus has been defined.

---

# 4. System Design

## 4.1 Architecture

The design contains four main modules:

| Module | Responsibility | Main file |
| --- | --- | --- |
| Morse generator | Defines the alphabet, creates timed tone signals, adds noise, and calculates spectrograms. | `morse.py` |
| Model and training | Defines the neural network, CTC conversion logic, dataset, batching, loss, optimizer, and checkpoint saving. | `main.py` |
| Decoder | Loads WAV input and model weights, preprocesses audio, runs inference, and returns text. | `decode_audio.py` |
| Desktop UI | Selects files, calls the decoder, displays status/errors, and saves output. | `gui.py` |

## 4.2 Input design

The inference input is a WAV file. If the input contains more than one audio channel, the channels are averaged to create mono audio. The audio is resampled to the project sampling frequency of 2,000 Hz and normalized by its peak magnitude. Empty and silent input are rejected with meaningful errors.

For training, the generator selects a random text sequence and constructs its waveform according to the Morse dictionary. It uses the standard PARIS reference convention to calculate dot duration from WPM. A dash is approximately three dots; symbol, character, and word spacing follow Morse timing relationships. Gaussian variation is applied to symbol duration, and noise is added to simulate imperfect transmissions.

## 4.3 Feature design

The system uses `scipy.signal.spectrogram` with a 40-sample (20 ms at 2 kHz) non-overlapping window. The output is a matrix of frequency bins over time. Before entering the network, this matrix is transposed to the shape:

```text
[time steps, spectral feature bins]
```

This representation allows the model to make a prediction at every time step while retaining information about the tone frequency and temporal structure.

## 4.4 Model design

The neural model is implemented in PyTorch. Its architecture is:

| Layer | Configuration |
| --- | --- |
| Dense 1 | Spectrogram size → 256, ReLU |
| Dense 2 | 256 → 256, ReLU |
| Dense 3 | 256 → 256, ReLU |
| Dense 4 | 256 → 256, ReLU |
| LSTM | 256 hidden units, batch-first sequence input |
| Output dense layer | 256 → number of classes plus CTC blank |
| Output activation | Log softmax across classes |

The target alphabet contains a space plus Morse-supported letters, digits, and punctuation. Index `0` is reserved for the CTC blank token. The training loop uses Adam optimization with an initial learning rate of `1e-3` and `CTCLoss`. Variable-length sequences are padded in batches, while their original input and target lengths are retained for the loss calculation.

## 4.5 Output design

During inference, the decoder selects the most likely class at each time frame. The `prediction_to_str` function then performs greedy CTC decoding: consecutive duplicate indices are collapsed, blank indices are removed, and the remaining indices are mapped back to characters. The command-line tool prints the prediction by default or saves it when `--output` is supplied. The GUI writes the same decoded text to the selected `.txt` file.

## 4.6 Data-flow design

The design follows a one-way data flow during decoding. Audio enters the system only through a selected file. The preprocessing stage produces a normalised waveform and does not modify the source audio. Feature extraction produces an in-memory spectrogram. The model produces an in-memory prediction sequence. Finally, the caller decides whether the textual result is displayed or written to disk. This flow keeps the original input intact and makes intermediate responsibilities easy to test independently.

```text
User → WAV path → audio reader → preprocessing → spectrogram → neural model
     ← status   ← error handler  ← validation    ← CTC decoder ← class sequence
                                                   ↓
                                           text display / output file
```

The training flow differs because it creates data repeatedly. A dataset request samples a new random configuration instead of reading a fixed training record. The collate function pads multiple generated feature sequences to a common length. Original lengths accompany the padded tensors so that CTC ignores padding. The loss calculated for one batch updates the model parameters, and periodic checkpoints preserve the state for later inference.

## 4.7 Interface design rationale

The GUI is intentionally compact. It contains only the controls required for the common task: an audio-file field, an output-file field, browse buttons, a Decode button, and a status area. The audio file chooser filters for WAV files while still permitting all files if necessary. When a user selects an audio file and has not specified an output path, the application proposes a text path with the same base filename. This small automation reduces repetitive work but does not remove user choice.

Visual design uses a restrained palette and clear labels. The Decode button is disabled during processing, which prevents two concurrent decode operations from writing the same output or making the interface appear unresponsive. Success messages state where the file was saved. Errors are shown in a dialog and through a status update. This is an appropriate interaction model for a short, file-based operation.

## 4.8 Algorithm design

The following pseudocode summarises inference:

```text
function decode_audio(path, model_path):
    rate, samples ← read WAV(path)
    if samples have multiple channels: average channels
    if samples are empty: raise error
    samples ← resample(samples, 2000 Hz)
    samples ← convert to floating point
    if peak absolute value is zero: raise error
    samples ← samples / peak absolute value
    features ← spectrogram(samples, 20 ms non-overlapping windows)
    model ← create network and load model_path on CPU
    logits ← model(features)
    labels ← argmax(logits at every frame)
    return collapse adjacent duplicate labels, remove blanks, map labels to characters
```

The generator uses an analogous algorithm in reverse: it transforms known characters into Morse elements and an audio waveform. A dot duration is calculated from WPM using the conventional PARIS reference. A dash is generated at roughly three dot durations. Appropriate zero-valued gaps are inserted between signs, characters, and words. The carrier sine wave makes the envelope audible, and noise is added before feature extraction.

## 4.9 Design quality attributes

**Modularity:** Morse generation, network definition, inference, and user interface are stored in distinct files. This allows the same `decode_audio` function to serve both interfaces.

**Maintainability:** Constants such as sampling frequency, alphabet, and default model path are named in code. A replacement model can be supplied through the command-line `--model` option without editing source.

**Portability:** Inference selects CPU execution and uses cross-platform Python libraries. GUI availability depends on Tkinter and the required image support.

**Testability:** The generator can emit a waveform, spectrogram, and known label; the decoder can be called as a Python function; and supplied audio artefacts provide a starting point for repeatable tests. A future revision should add automated unit and integration tests.

---

# 5. Implementation

## 5.1 Development environment

The project is implemented in Python. The dependency definition in `Pipfile` lists PyTorch, SciPy, NumPy, TensorBoard, Matplotlib, and Pillow; Tkinter is required for the desktop interface. The project specifies Python 3.7, though dependency compatibility should be checked when using a newer Python release.

## 5.2 Training implementation

`Dataset` in `main.py` creates 2,048 synthetic samples per dataset pass. Each sample varies the following values:

| Parameter | Training range in implementation |
| --- | --- |
| Text length | 10–19 characters |
| Pitch | 100–949 Hz |
| Speed | 10–39 WPM |
| Noise control value | 0–199 |
| Amplitude control value | 10–149 |

The data loader groups 64 examples into a batch. The training process logs loss to TensorBoard and saves model weights every ten epochs as `models/<epoch>.pt`. The training script configures a CUDA device, so a CUDA-capable PyTorch installation is needed to run the training loop as currently written.

## 5.3 Decoding implementation

The standard usage is:

```sh
python3 decode_audio.py audio/hello_world.wav
python3 decode_audio.py audio/hello_world.wav --output decoded.txt
```

The default model path is resolved relative to the project directory, so the bundled `models/001750.pt` checkpoint is used unless the `--model` option specifies another file.

## 5.4 GUI implementation

Run the interface with:

```sh
python3 gui.py
```

The interface provides audio and output-file selectors, a Decode button, a progress status message, and error dialogs. It disables the Decode button while inference is running to prevent duplicate actions. The output directory is created when necessary before the result is written.

## 5.5 Repository artefacts

| Artefact | Purpose |
| --- | --- |
| `audio/hello_world.wav` | Clean example input audio |
| `audio/hello_world_noise.wav` | Noisy example input audio |
| `audio/hello_world.txt` | Reference text for the example |
| `models/001750.pt` | Bundled trained model checkpoint |
| `hello_world.png` | Example project output image |
| `logo.jpeg` | GUI logo image |

## 5.6 Detailed preprocessing implementation

The decoder performs preprocessing in a deliberate order. First, `scipy.io.wavfile.read` reads the sample rate and waveform. Stereo or multichannel data is reduced by calculating the mean across channels. This gives the model one consistent input shape. Next, the duration of the source recording is calculated from the source sample rate, and `scipy.signal.resample` creates the corresponding number of samples at 2 kHz. Resampling is necessary because the generator and feature extractor are designed around that project sample rate.

The resampled data is converted to `float32` and divided by its maximum absolute value. Normalisation reduces sensitivity to the scale used by a recorder or WAV encoding. It does not remove all amplitude-related difficulty, but it keeps the input range stable. A zero peak indicates silent audio and is rejected before division. The feature matrix is then converted to a PyTorch tensor, transposed so time is the leading sequence dimension, and expanded with a batch dimension before model inference.

## 5.7 Detailed model-training implementation

Each generated text label is mapped to integer indices using `tag_to_idx`; zero is intentionally excluded because CTC reserves it as blank. The collate function pads feature sequences and target sequences independently. It also records the true time length of every feature sequence and true character length of every label. These lengths are passed to PyTorch's `CTCLoss`, ensuring that padded positions do not contribute to the learning objective.

The forward method applies four linear transformations with ReLU activation. ReLU is computationally simple and introduces non-linearity, allowing the model to learn patterns beyond a linear frequency threshold. The LSTM then reads the sequence in batch-first form and produces a hidden representation for each frame. The final dense layer projects each hidden representation to class scores, and log softmax supplies log-probabilities expected by the CTC loss. During training, output dimensions are permuted because PyTorch CTC expects time-major logits.

The training loop zeroes gradients, performs the forward pass, calculates loss, back-propagates gradients, and updates weights through Adam. Loss is recorded with TensorBoard once per epoch. Checkpoints are saved at ten-epoch intervals, enabling model recovery and selection of a suitable training state. The source currently begins with epoch zero unless manually changed, so a production-quality extension should add command-line configuration for resume checkpoint, epoch count, seed, device, and output directory.

## 5.8 Installation and execution procedure

1. Install a compatible Python version and the packages listed in `Pipfile`.
2. Ensure SciPy and PyTorch are available for decoding. Install Tkinter and Pillow if the GUI will be used.
3. Keep `models/001750.pt` in the repository's `models` directory, or provide a replacement via `--model`.
4. Run `python3 decode_audio.py <input.wav>` for terminal output.
5. Add `--output <result.txt>` to save the prediction.
6. Run `python3 gui.py` to use the desktop interface.
7. To train a new model, run `python3 main.py` on a machine with a CUDA-enabled PyTorch setup and monitor TensorBoard output.

Because package versions are intentionally broad in the supplied dependency file, a reproducible deployment should create and record a tested lockfile or requirements file for the target operating system. The actual availability of CUDA, Tkinter, and audio-library dependencies should be verified on the deployment machine.

## 5.9 Error handling and operational behaviour

The code handles several common failure conditions. The decoder raises a `ValueError` for an empty audio array and for a silent waveform. The GUI checks that an audio path has been selected and that it points to an existing file before calling the decoder. Its decode operation is wrapped in `try/except/finally`: errors change the status text and appear in a dialog, while the `finally` block always restores the Decode button. This is a sound pattern because the user is not left with a permanently disabled interface after an exception.

The command-line program leaves unexpected errors visible to the terminal, which is appropriate for a developer-oriented utility but could be improved by adding friendly exit messages and error codes. Input type validation, model compatibility checks, logging, and automated tests are recommended enhancements. Importantly, the system should never present an unverified prediction as a guaranteed transcription; output should be interpreted in the context of input quality and model evaluation.

## 5.10 Security and ethical considerations

The application operates on local files and does not transmit audio or decoded text to an external service. This is a positive privacy property for recordings that may be sensitive. Users should still obtain permission before processing recordings belonging to others and should comply with local laws and organisational policies governing radio communications. The decoder is an assistive transcription tool, not a substitute for verification in safety-critical or legal contexts. Its limitations should be communicated honestly, especially where signal quality is poor or the supported alphabet is insufficient.

---

# 6. Results and Conclusion

## 6.1 Results

The repository includes clean and noisy “hello world” WAV examples, a corresponding reference text file, a saved trained model, and an output illustration. These artefacts demonstrate the intended end-to-end workflow: audio is supplied to the decoder, transformed into spectrogram frames, processed by the neural network, and converted to text.

The implementation is designed to improve robustness by training across varied pitch, speed, noise, amplitude, and timing. However, this repository does not include a formal held-out test set, character error rate (CER), word error rate (WER), or benchmark log. Therefore, this report does **not** claim a numerical accuracy value. A reproducible evaluation should compare decoder output with known reference text over a separate clean/noisy test corpus and report:

```text
CER = (substitutions + deletions + insertions) / number of reference characters
WER = (word substitutions + word deletions + word insertions) / number of reference words
```

## 6.2 Limitations

- Inference uses greedy decoding; the source code explicitly notes that proper beam search remains to be implemented.
- The model is trained on synthetic audio, so its performance on recordings from real radios, microphones, and senders should be measured separately.
- The current training program assumes CUDA rather than automatically falling back to CPU.
- Input is limited to WAV files, and the decoder does not include live microphone capture.
- No language model is used to correct unlikely character sequences.

## 6.3 Future enhancements

- Add a labelled real-world Morse dataset and report CER/WER across different noise levels and speeds.
- Implement beam-search CTC decoding and optionally integrate a language model.
- Add CPU fallback and configurable training parameters.
- Support microphone streaming and real-time incremental text display.
- Add audio-format conversion and an interface for adjusting preprocessing settings.
- Compare the neural approach with a classical tone-duration baseline under identical test conditions.

## 6.4 Conclusion

This project demonstrates an end-to-end method for decoding Morse-code audio using deep learning. It replaces hand-tuned dot/dash segmentation with a spectrogram-to-text model composed of dense layers, an LSTM, and CTC training. Synthetic data generation exposes the model to changes in speed, frequency, amplitude, noise, and timing. The saved model, command-line tool, and desktop GUI make the result usable as a local application. The main next step is rigorous evaluation on held-out and real recordings so that its accuracy and robustness can be quantified.

## 6.5 Evaluation plan

A credible final evaluation must separate the data used to develop the system from the data used to judge it. The recommended procedure is to create three disjoint sets: training data for parameter updates, validation data for model and decoder choices, and test data for the final report. The test set should contain a balanced selection of message lengths, characters, speeds, carrier frequencies, amplitude levels, clean recordings, and controlled noise conditions. Where possible, it should also include recordings from real senders and devices not represented in training.

For each test item, retain the exact reference transcription and capture the decoder output. Calculate character error rate and word error rate, then report the mean and distribution of errors rather than only a single favourable example. A results table may be structured as follows once measurements are available:

| Condition | Number of files | Mean CER | Mean WER | Notes |
| --- | ---: | ---: | ---: | --- |
| Clean synthetic audio | [fill after test] | [value] | [value] | Same parameter ranges as generator |
| Noisy synthetic audio | [fill after test] | [value] | [value] | State noise level/range |
| Real recorded audio | [fill after test] | [value] | [value] | State device and environment |
| Unseen WPM range | [fill after test] | [value] | [value] | Tests speed generalisation |

This method prevents a common reporting error: treating a successful example as a measure of overall performance. The supplied repository demonstrates functional readiness, but results should only be inserted into the table after running the stated experiment. Screenshots of the GUI and terminal output can accompany the measured table in a final printed submission.

## 6.6 Interpretation of expected behaviour

The system should work best when a recording contains a clear, single Morse tone whose speed, pitch, and noise conditions resemble the diversity used in the generator. Peak normalisation and frequency features help with ordinary level variation. The LSTM can use the progression of frames to distinguish active tone from silence and infer character-level patterns. The random timing variation in generated samples is intended to make it less brittle than a decoder trained only on perfectly regular code.

Errors are more likely when audio is severely clipped, nearly silent, dominated by interference, transmitted at very unusual speeds, or contains a carrier outside the learned range. The greedy decoder may also make a locally plausible but globally poor decision because it considers only the most likely class at each frame. A beam search can retain several candidate sequences and may improve accuracy, especially in ambiguous regions. A language model could further prefer probable word sequences, but it must be carefully assessed to avoid “correcting” valid specialised messages into incorrect common words.

## 6.7 Lessons learned

Several engineering lessons emerge from the project. First, data quality and diversity are as important as neural architecture. A small model can perform a meaningful task when the generator encodes the essential variations, but it cannot learn conditions that are absent from its training examples. Second, an end-to-end model simplifies the visible pipeline but does not eliminate the need for good preprocessing. Sample-rate consistency, normalisation, and correct tensor dimensions are essential for a model trained on spectral input.

Third, a deployable project includes more than a trained network. The command-line interface makes automation possible, while the GUI makes demonstration and non-technical use easier. Fourth, transparency matters. The repository's explicit TODO for beam search and absence of a reported benchmark should be treated as opportunities for future work, not concealed. An academically sound conclusion describes what was demonstrated, what was not measured, and how the next experiment will answer the remaining questions.

## 6.8 Final conclusion

The completed system meets its principal design objective: it provides a local pipeline that can transform Morse-code WAV audio into a predicted text sequence using a trained neural model. It integrates data synthesis, spectral feature extraction, dense neural processing, recurrent temporal modelling, CTC-based sequence learning, greedy decoding, command-line access, and a graphical interface. This integration is the key result of the project.

The work should be viewed as a well-defined foundation for a stronger decoder rather than as the final word on Morse recognition. Its next stage is empirical: collect a labelled evaluation corpus, measure CER and WER, compare against a timing-rule baseline, and refine data and decoding methods based on observed failures. With those additions, the project can progress from a functional proof of concept to a quantitatively validated communication-assistance tool.

---

# 7. Acknowledgement

I express my sincere gratitude to **[Guide name]** for valuable guidance, encouragement, and constructive feedback throughout this project. I thank the faculty members of **[Department name]**, **[Institution name]**, for providing the resources and academic environment needed to complete this work. I am also grateful to my classmates, friends, and family for their support and motivation. Finally, I acknowledge the developers and open-source communities behind Python, PyTorch, SciPy, and the other tools that made this project possible.

I would like to place on record my sincere appreciation for the opportunity to undertake this final-year project. The work offered an opportunity to bring together concepts that are often studied separately: digital communication, audio processing, data generation, neural networks, sequence learning, software engineering, and user-interface development. Completing a project of this nature requires patience, repeated experimentation, and the willingness to learn from results that are incomplete or unexpected. The guidance received during that process was invaluable.

I am deeply thankful to **[Guide name]**, project guide, for providing direction at every important stage of the work. The guide's suggestions helped refine the problem from a general idea about Morse code into a focused audio-to-text system with clear objectives. Advice on defining realistic scope, separating training from inference, documenting limitations, and presenting results responsibly improved both the technical quality and the academic clarity of this report. I am grateful for the time spent reviewing progress, identifying weaknesses, and encouraging a systematic approach to the project.

I extend my gratitude to the Head of the Department, faculty members, laboratory staff, and administrative staff of **[Department name]**, **[Institution name]**. Their support provided the academic setting in which this work could be carried out. The courses, discussions, facilities, and assessment process all contributed to the knowledge needed for this project. I especially acknowledge the importance of an environment that encourages students to experiment with modern tools while also maintaining sound engineering and documentation practices.

My sincere thanks go to my classmates and friends for their encouragement, feedback, and help during the development process. Peer discussion is particularly valuable in a software project because it challenges assumptions that may otherwise remain unnoticed. Conversations about model design, data quality, error cases, usability, and report structure helped improve the way the problem was understood and communicated. Their moral support during the demanding stages of implementation and documentation is gratefully acknowledged.

I am profoundly grateful to my parents and family members for their constant encouragement, patience, and confidence. Their support created the time and motivation required to complete the work. A final-year project is not only a technical exercise; it is also a sustained commitment that depends on the understanding of people around the student. I acknowledge their contribution with great respect and affection.

I also acknowledge the authors, maintainers, reviewers, and communities of the open-source software used in this work. Python made it possible to develop a clear and accessible implementation. PyTorch supplied the tools for building, training, and loading the neural model. SciPy and NumPy supported audio reading, resampling, spectrogram computation, and numerical processing. TensorBoard supported training observation, while Tkinter and Pillow enabled the desktop interface. The availability of these tools, their documentation, and the collective work of their contributors made this project practical within an academic setting.

Finally, I thank all researchers whose publications on LSTM networks, CTC, speech recognition, audio processing, and international Morse code informed the conceptual basis of this report. Academic work advances through the sharing of ideas, reproducible methods, and careful critique. This project has benefited from that tradition. Any limitations, omissions, or errors that remain in the implementation or report are my own responsibility. I hope that the work demonstrates sincere effort, contributes to learning, and provides a useful foundation for future improvement.

I further acknowledge the value of constructive evaluation in the completion of this project. Questions raised during reviews and demonstrations encouraged a clearer distinction between a working software feature and a fully measured research result. That distinction is important in engineering education. It guided the preparation of the results chapter, where the existence of example files and a bundled model is documented without inventing numerical accuracy claims. The feedback process reinforced the importance of test data, repeatability, error analysis, and honest communication of limitations. These lessons will remain useful beyond this specific project.

The project also benefited from the discipline of documentation. Recording the purpose of each module, the path of data through the system, the supported alphabet, and the assumptions behind preprocessing made the software easier to understand and maintain. Writing the report revealed areas where future work would be valuable, including a held-out evaluation corpus, beam-search decoding, CPU fallback during training, and live audio support. I am grateful for the academic requirement to document the work in detail because it transformed implementation decisions into learning outcomes that can be shared with others.

I would also like to thank everyone who contributed, directly or indirectly, to a culture of curiosity and responsible technology use during my studies. The subject of this project combines an older communication system with current machine-learning methods. It is a reminder that useful innovation often comes from revisiting familiar problems with new tools, while still respecting the knowledge and standards developed by earlier practitioners. This perspective has made the project both technically engaging and personally meaningful.

With sincere respect, I dedicate this work to all teachers, mentors, family members, friends, and fellow students who supported my academic journey. Their confidence and encouragement helped transform an initial idea into a completed project report and working software artefact. I remain thankful for their contribution and for the opportunity to learn through this experience.

---

# References

1. S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997. doi: 10.1162/neco.1997.9.8.1735.
2. A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, “Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks,” *Proceedings of the 23rd International Conference on Machine Learning*, pp. 369–376, 2006. doi: 10.1145/1143844.1143891.
3. A. Graves, A.-R. Mohamed, and G. Hinton, “Speech Recognition with Deep Recurrent Neural Networks,” *ICASSP*, pp. 6645–6649, 2013. doi: 10.1109/ICASSP.2013.6638947.
4. International Telecommunication Union, “Recommendation ITU-R M.1677-1: International Morse Code,” 2009.
