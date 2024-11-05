# Aurora Project

Aurora is a Speech-to-Text (STT), Language Model (LLM), and Text-to-Speech (TTS) integration project. The project aims to create a natural, conversational AI experience by utilizing voice input, processing it through a language model, and returning spoken responses.

## Features

- **STT**: Converts spoken input to text.
- **LLM**: Processes the text input using an advanced language model to generate responses.
- **TTS**: Converts the generated text response back into speech output.

## Requirements

- Python 3.8 or higher
- Required libraries: `torch`, `pygame`, `sounddevice`, `gTTS`, `vosk`, `transformers`

## Model Download

For the STT component, you need to download the Vosk model from the following link:

[Vosk Models Download](https://alphacephei.com/vosk/models)

## Installation

1. Clone the repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Download the Vosk model and place it in the appropriate directory as specified in the code.

## Usage

Run the main script to initiate the conversation process. Speak into the microphone, and the system will respond with an appropriate answer.

## License

This project is licensed under a permissive license inspired by MIT, which encourages creativity and innovation. You are free to use, clone, modify, and distribute this code for both non-commercial and commercial purposes, provided that appropriate credit is given to the original author. Any derivative works must maintain the same open and permissive spirit, ensuring that they can also be freely used, modified, and distributed by others.

